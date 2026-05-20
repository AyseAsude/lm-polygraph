import torch
import openai
import time
import random
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from dataclasses import asdict, dataclass
from typing import List, Dict, Optional, Union
from abc import abstractmethod, ABC
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig,
    LogitsProcessorList,
    BartForConditionalGeneration,
)
from huggingface_hub import InferenceClient

from lm_polygraph.utils.generation_parameters import (
    GenerationParameters,
    GenerationParametersFactory,
)
from lm_polygraph.utils.ensemble_utils.ensemble_generator import EnsembleGenerationMixin
from lm_polygraph.utils.ensemble_utils.dropout import replace_dropout

log = logging.getLogger("lm_polygraph")


@dataclass
class TopLogprobWrapper:
    """Wrapper to match OpenAI's TopLogprob structure."""
    token: str
    logprob: float


@dataclass
class TokenLogprobWrapper:
    """Wrapper to match OpenAI's ChatCompletionTokenLogprob structure."""
    token: str
    logprob: float
    top_logprobs: List["TopLogprobWrapper"]


@dataclass
class LogprobsWrapper:
    """Wrapper to match OpenAI's ChoiceLogprobs structure."""
    content: List[TokenLogprobWrapper]


# Transient errors worth retrying. Other API errors (auth, bad request)
# propagate immediately so we fail fast on real problems.
_RETRYABLE_API_EXCEPTIONS = (
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIConnectionError,
    openai.APITimeoutError,
)


def _retry_delay_seconds(exception, attempt: int) -> float:
    """Honor the provider's Retry-After header if present, else exponential backoff with jitter."""
    response = getattr(exception, "response", None)
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after is not None:
            try:
                return float(retry_after)
            except (TypeError, ValueError):
                pass
    return min(2 ** attempt, 30.0) + random.uniform(0, 1)


def _normalize_logprobs(logprobs_obj):
    """
    Convert Together AI logprobs format to OpenAI-compatible format.

    Together AI format:
        - content: None
        - tokens: ['The', ' city', ...]
        - token_logprobs: [-0.026, -0.0002, ...]
        - top_logprobs: [{'The': -0.026, 'New': -3.65}, ...]

    OpenAI format:
        - content: [TokenLogprobWrapper(...), ...]
    """
    if logprobs_obj is None:
        return None

    # Already OpenAI format - content is a non-None list
    if hasattr(logprobs_obj, "content") and logprobs_obj.content is not None:
        return logprobs_obj

    # Together AI format - convert to OpenAI format
    if hasattr(logprobs_obj, "tokens") and logprobs_obj.tokens is not None:
        content = []
        tokens = logprobs_obj.tokens
        token_logprobs = logprobs_obj.token_logprobs or []
        top_logprobs_list = getattr(logprobs_obj, "top_logprobs", None) or []

        for i, token in enumerate(tokens):
            # Get logprob for this token (default to 0.0 if missing)
            logprob = token_logprobs[i] if i < len(token_logprobs) else 0.0

            # Convert top_logprobs dict to list of TopLogprobWrapper
            top_lps = []
            if i < len(top_logprobs_list) and top_logprobs_list[i]:
                for t, lp in top_logprobs_list[i].items():
                    top_lps.append(TopLogprobWrapper(token=t, logprob=lp))

            content.append(TokenLogprobWrapper(
                token=token,
                logprob=logprob,
                top_logprobs=top_lps
            ))

        return LogprobsWrapper(content=content)

    # Unknown format - return as-is
    return logprobs_obj


class Model(ABC):
    """
    Abstract model class. Used as base class for both White-box models and Black-box models.
    """

    def __init__(self, model_path: str, model_type: str):
        """
        Parameters:
            model_path (str): unique model path where it can be found.
            model_type (str): description of additional model properties. Can be 'Blackbox' or model specifications
                in the case of white-box.
        """
        self.model_path = model_path
        self.model_type = model_type

    @abstractmethod
    def generate_texts(self, input_texts: List[str], **args) -> List[str]:
        """
        Abstract method. Generates a list of model answers using input texts batch.

        Parameters:
            input_texts (List[str]): input texts batch.
        Return:
            List[str]: corresponding model generations. Have the same length as `input_texts`.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def generate(self, **args):
        """
        Abstract method. Generates the model output with scores from batch formed by HF Tokenizer.
        Not implemented for black-box models.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def __call__(self, **args):
        """
        Abstract method. Calls the model on the input batch. Returns the resulted scores.
        Not implemented for black-box models.
        """
        raise Exception("Not implemented")


class BlackboxModel(Model):
    """
    Black-box model class. Have no access to model scores and logits.
    Currently implemented blackbox models: OpenAI models, Huggingface models.

    Examples:

    ```python
    >>> from lm_polygraph import BlackboxModel
    >>> model = BlackboxModel.from_openai(
    ...     'YOUR_OPENAI_TOKEN',
    ...     'gpt-3.5-turbo'
    ... )
    ```

    ```python
    >>> from lm_polygraph import BlackboxModel
    >>> model = BlackboxModel.from_huggingface(
    ...     hf_api_token='YOUR_API_TOKEN',
    ...     hf_model_id='google/t5-large-ssm-nqo'
    ... )
    ```
    """

    def __init__(
        self,
        openai_api_key: str = None,
        model_path: str = None,
        hf_api_token: str = None,
        generation_parameters: GenerationParameters = GenerationParameters(),
        supports_logprobs: bool = False,
        base_url: str = None,
    ):
        """
        Parameters:
            openai_api_key (Optional[str]): OpenAI API key if the blackbox model comes from OpenAI. Default: None.
            model_path (Optional[str]): Unique model path. Openai model name, if `openai_api_key` is specified,
                huggingface path, if `hf_api_token` is specified. Default: None.
            hf_api_token (Optional[str]): Huggingface API token if the blackbox model comes from HF. Default: None.
            generation_parameters (GenerationParameters): parameters to use in model generation. Default: default parameters.
            supports_logprobs (bool): Whether the model supports returning log probabilities. Default: False.
            base_url (Optional[str]): Base URL for OpenAI-compatible API providers. If None, uses the default
                OpenAI API endpoint. Default: None.
        """
        super().__init__(model_path, "Blackbox")
        self.generation_parameters = generation_parameters
        self.openai_api_key = openai_api_key
        self.supports_logprobs = supports_logprobs
        self.base_url = base_url

        if openai_api_key is not None:
            self.openai_api = openai.OpenAI(api_key=openai_api_key, base_url=base_url)

        self.hf_api_token = hf_api_token

    def _validate_args(self, args):
        """
        Validates and adapts arguments for BlackboxModel generation.

        Parameters:
            args (dict): The arguments to validate.

        Returns:
            dict: Validated and adapted arguments.
        """
        args_copy = args.copy()

        # When pointed at an OpenAI-compatible provider (e.g. Together AI),
        # forward provider-specific params via the SDK's extra_body escape hatch
        # so they survive the SDK's typed signature check.
        passthrough_for_custom_base_url = ["repetition_penalty", "top_k"]
        extra_body = {}
        if self.base_url is not None:
            for key in passthrough_for_custom_base_url:
                if key in args_copy and args_copy[key] is not None:
                    extra_body[key] = args_copy.pop(key)

        # BlackboxModel specific validation
        for delete_key in [
            "do_sample",
            "min_length",
            "top_k",
            "repetition_penalty",
            "min_new_tokens",
            "num_beams",
            "allow_newlines",
            "stop_strings",
        ]:
            args_copy.pop(delete_key, None)

        if extra_body:
            args_copy["extra_body"] = extra_body

        # Map HF argument names to OpenAI/HF API argument names
        key_mapping = {
            "num_return_sequences": "n",
            "max_length": "max_tokens",
            "max_new_tokens": "max_tokens",
        }
        for key, replace_key in key_mapping.items():
            if key in args_copy:
                args_copy[replace_key] = args_copy[key]
                args_copy.pop(key)

        return args_copy

    def _query(self, payload):
        client = InferenceClient(model=self.model_path, token=self.hf_api_token)
        response = client.chat_completion(payload)
        raw_json = json.dumps(response, indent=2)
        return raw_json

    @staticmethod
    def from_huggingface(hf_api_token: str, hf_model_id: str, **kwargs):
        """
        Initializes a blackbox model from huggingface.

        Parameters:
            hf_api_token (Optional[str]): Huggingface API token if the blackbox model comes from HF. Default: None.
            hf_model_id (Optional[str]): model path in huggingface.
        """
        generation_parameters = kwargs.pop(
            "generation_parameters", GenerationParameters()
        )
        return BlackboxModel(
            hf_api_token=hf_api_token,
            model_path=hf_model_id,
            generation_parameters=generation_parameters,
        )

    @staticmethod
    def from_openai(
        openai_api_key: str, model_path: str, supports_logprobs: bool = False, base_url: str = None, **kwargs
    ):
        """
        Initializes a blackbox model from OpenAI API.

        Parameters:
            openai_api_key (Optional[str]): OpenAI API key. Default: None.
            model_path (Optional[str]): model name in OpenAI.
            supports_logprobs (bool): Whether the model supports returning log probabilities. Default: False.
        """
        generation_parameters = kwargs.pop(
            "generation_parameters", GenerationParameters()
        )
        return BlackboxModel(
            openai_api_key=openai_api_key,
            model_path=model_path,
            supports_logprobs=supports_logprobs,
            generation_parameters=generation_parameters,
            base_url=base_url,
        )

    def generate_texts(self, input_texts: List[str], **args) -> List[str]:
        """
        Generates a list of model answers using input texts batch.

        Parameters:
            input_texts (List[str]): input texts batch.
        Return:
            List[str]: corresponding model generations. Have the same length as `input_texts`.
        """
        # Apply default parameters first, then override with provided args
        default_params = asdict(self.generation_parameters)
        default_params.update(args)
        args = self._validate_args(default_params)
        max_parallel_requests = args.pop("max_parallel_requests", None)
        if max_parallel_requests is None:
            max_parallel_requests = min(2, len(input_texts))
        else:
            max_parallel_requests = int(max_parallel_requests)
            if max_parallel_requests < 1:
                raise ValueError("max_parallel_requests must be >= 1")

        # Check if we're trying to access features that require logprobs support
        if (
            any(
                args.get(arg, False)
                for arg in [
                    "output_scores",
                    "output_attentions",
                    "output_hidden_states",
                ]
            )
            and not self.supports_logprobs
        ):
            raise Exception("Cannot access logits for blackbox model")

        texts = []

        if self.openai_api_key is not None:
            # Save log probabilities if requested
            self.last_response = None
            self.logprobs = []
            self.tokens = []

            # Check if we need to return logprobs
            return_logprobs = args.pop("output_scores", False)
            logprobs_args = {}

            if return_logprobs and self.supports_logprobs:
                logprobs_args["logprobs"] = True
                # OpenAI supports returning top logprobs, default to 5
                logprobs_args["top_logprobs"] = args.pop("top_logprobs", 5)

            def _prepare_messages(prompt):
                if isinstance(prompt, str):
                    # If prompt is a string, create a single message with "user" role
                    return [{"role": "user", "content": prompt}]
                elif isinstance(prompt, list) and all(
                    isinstance(item, dict) for item in prompt
                ):
                    # If prompt is a list of dictionaries, assume it's already structured as chat
                    return prompt
                else:
                    raise ValueError(
                        "Invalid prompt format. Must be either a string or a list of dictionaries."
                    )

            def _generate_single(index, prompt):
                messages = _prepare_messages(prompt)
                max_retries = 5
                for attempt in range(max_retries + 1):
                    try:
                        response = self.openai_api.chat.completions.create(
                            model=self.model_path,
                            messages=messages,
                            **args,
                            **logprobs_args,
                        )
                        break
                    except _RETRYABLE_API_EXCEPTIONS as e:
                        if attempt >= max_retries:
                            raise
                        delay = _retry_delay_seconds(e, attempt)
                        log.warning(
                            "API request failed (%s), retrying in %.1fs (attempt %d/%d)",
                            type(e).__name__, delay, attempt + 1, max_retries,
                        )
                        time.sleep(delay)

                if args.get("n", 1) == 1:
                    text = response.choices[0].message.content
                    normalized_logprobs = None
                    token_list = None
                    has_logprobs = False
                    # Store logprobs if available
                    if return_logprobs and hasattr(response.choices[0], "logprobs"):
                        has_logprobs = True
                        # Normalize logprobs to OpenAI format (handles Together AI)
                        normalized_logprobs = _normalize_logprobs(response.choices[0].logprobs)
                        # Extract token information if available
                        if normalized_logprobs is not None and hasattr(normalized_logprobs, "content") and normalized_logprobs.content is not None:
                            token_list = [item.token for item in normalized_logprobs.content]
                else:
                    text = [resp.message.content for resp in response.choices]
                    normalized_logprobs = None
                    token_list = None
                    has_logprobs = False
                    # For multiple returns, we don't collect logprobs for now

                return index, text, normalized_logprobs, token_list, has_logprobs, response

            indexed_texts = [None] * len(input_texts)
            indexed_logprobs = [None] * len(input_texts)
            indexed_tokens = [None] * len(input_texts)
            indexed_has_logprobs = [False] * len(input_texts)
            indexed_responses = [None] * len(input_texts)
            max_workers = min(max_parallel_requests, len(input_texts))

            if max_workers <= 1:
                for i, prompt in enumerate(input_texts):
                    idx, text, normalized_logprobs, token_list, has_logprobs, response = _generate_single(i, prompt)
                    indexed_texts[idx] = text
                    indexed_logprobs[idx] = normalized_logprobs
                    indexed_tokens[idx] = token_list
                    indexed_has_logprobs[idx] = has_logprobs
                    indexed_responses[idx] = response
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(_generate_single, i, prompt)
                        for i, prompt in enumerate(input_texts)
                    ]
                    for future in as_completed(futures):
                        idx, text, normalized_logprobs, token_list, has_logprobs, response = future.result()
                        indexed_texts[idx] = text
                        indexed_logprobs[idx] = normalized_logprobs
                        indexed_tokens[idx] = token_list
                        indexed_has_logprobs[idx] = has_logprobs
                        indexed_responses[idx] = response

            texts.extend(indexed_texts)

            if return_logprobs and args.get("n", 1) == 1:
                for has_logprobs, normalized_logprobs, token_list in zip(
                    indexed_has_logprobs,
                    indexed_logprobs,
                    indexed_tokens,
                ):
                    if has_logprobs:
                        self.logprobs.append(normalized_logprobs)
                        if token_list is not None:
                            self.tokens.append(token_list)

            if indexed_responses:
                # Preserve previous semantics: last response corresponds to last input.
                self.last_response = indexed_responses[-1]

        elif (self.hf_api_token is not None) & (self.model_path is not None):
            for prompt in input_texts:
                start = time.time()
                while True:
                    current_time = time.time()
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ]
                    output = self._query(messages)

                    if isinstance(output, dict):
                        if (list(output.keys())[0] == "error") & (
                            "estimated_time" in output.keys()
                        ):
                            estimated_time = float(output["estimated_time"])
                            elapsed_time = current_time - start
                            print(
                                f"{output['error']}. Estimated time: {round(estimated_time - elapsed_time, 2)} sec."
                            )
                            time.sleep(5)
                        elif (list(output.keys())[0] == "error") & (
                            "estimated_time" not in output.keys()
                        ):
                            log.error(f"{output['error']}")
                            break
                    elif isinstance(output, list):
                        break

                texts.append(output[0]["generated_text"])
        else:
            print(
                "Please provide HF API token and model id for using models from HF or openai API key for using OpenAI models"
            )

        return texts

    def generate(self, **args):
        """
        For OpenAI models with logprobs support, returns a lightweight wrapper around OpenAI API response.
        For other blackbox models, raises an exception as this is not implemented.

        Parameters:
            **args: Arguments to pass to the generate method.
        Returns:
            object: A wrapper around the OpenAI API response if logprobs are supported.
        Raises:
            Exception: If the model doesn't support logprobs.
        """
        if self.supports_logprobs:
            # Apply default parameters first, then override with provided args
            default_params = asdict(self.generation_parameters)
            default_params.update(args)
            args = self._validate_args(default_params)

            args["output_scores"] = True
            sequences = self.generate_texts(**args)

            # Return a simple object with the necessary attributes for compatibility
            class OpenAIGenerationOutput:
                def __init__(self, sequences, scores):
                    self.sequences = sequences
                    self.scores = scores

            return OpenAIGenerationOutput(sequences, self.logprobs)
        else:
            raise Exception("Cannot access logits of blackbox model")

    def __call__(self, **args):
        """
        Not implemented for blackbox models.
        """
        raise Exception("Cannot access logits of blackbox model")

    def tokenizer(self, *args, **kwargs):
        """
        Not implemented for blackbox models.
        """
        raise Exception("Cannot access logits of blackbox model")


class WhiteboxModel(Model):
    """
    White-box model class. Have access to model scores and logits. Currently implemented only for Huggingface models.

    Examples:

    ```python
    >>> from lm_polygraph import WhiteboxModel
    >>> model = WhiteboxModel.from_pretrained(
    ...     "bigscience/bloomz-3b",
    ... )
    ```
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        model_path: str = None,
        model_type: str = "CausalLM",
        generation_parameters: GenerationParameters = GenerationParameters(),
        instruct: bool = False,
    ):
        """
        Parameters:
            model (AutoModelForCausalLM): HuggingFace model.
            tokenizer (AutoTokenizer): HuggingFace tokenizer.
            model_path (Optional[str]): Unique model path in HuggingFace.
            model_type (str): Additional model specifications.
            parameters (GenerationParameters): parameters to use in model generation. Default: default parameters.
        """
        super().__init__(model_path, model_type)
        self.model = model
        self.tokenizer = tokenizer
        self.generation_parameters = generation_parameters
        self.instruct = instruct

    def _validate_args(self, args):
        """
        Validates and adapts arguments for WhiteboxModel generation.

        Parameters:
            args (dict): The arguments to validate.

        Returns:
            dict: Validated and adapted arguments.
        """
        args_copy = args.copy()

        # WhiteboxModel specific validation
        if "presence_penalty" in args_copy and args_copy["presence_penalty"] != 0.0:
            log.warning(
                "Skipping requested argument presence_penalty={}".format(
                    args_copy["presence_penalty"]
                )
            )

        # Remove arguments that are not supported by the HF model.generate function
        keys_to_remove = ["presence_penalty", "allow_newlines"]
        for key in keys_to_remove:
            args_copy.pop(key, None)

        return args_copy

    class _ScoresProcessor:
        # Stores original token scores instead of the ones modified with generation parameters
        def __init__(self):
            self.scores = []

        def __call__(self, input_ids=None, scores=None):
            self.scores.append(scores.log_softmax(-1))
            return scores

    def generate(self, **args):
        """
        Generates the model output with scores from batch formed by HF Tokenizer.

        Parameters:
            **args: Any arguments that can be passed to model.generate function from HuggingFace.
        Returns:
            ModelOutput: HuggingFace generation output with scores overriden with original probabilities.
        """
        default_params = asdict(self.generation_parameters)

        # add ScoresProcessor to collect original scores
        processor = self._ScoresProcessor()
        if "logits_processor" in args.keys():
            logits_processor = LogitsProcessorList(
                [processor, args["logits_processor"]]
            )
        else:
            logits_processor = LogitsProcessorList([processor])
        args["logits_processor"] = logits_processor

        # update default parameters with passed arguments
        default_params.update(args)
        args = default_params

        if "stop_strings" in args:
            args["tokenizer"] = self.tokenizer

        args = self._validate_args(args)
        generation = self.model.generate(**args)

        # override generation.scores with original scores from model
        generation.generation_scores = generation.scores
        generation.scores = processor.scores

        return generation

    def generate_texts(self, input_texts: List[str], **args) -> List[str]:
        """
        Generates a list of model answers using input texts batch.

        Parameters:
            input_texts (List[str]): input texts batch.
        Return:
            List[str]: corresponding model generations. Have the same length as `input_texts`.
        """
        # Apply default parameters first, then override with provided args
        default_params = asdict(self.generation_parameters)
        default_params.update(args)
        args = self._validate_args(default_params)

        args["return_dict_in_generate"] = True
        batch: Dict[str, torch.Tensor] = self.tokenize(input_texts)
        batch = {k: v.to(self.device()) for k, v in batch.items()}
        sequences = self.generate(**batch, **args).sequences.cpu()
        input_len = batch["input_ids"].shape[1]
        texts = []

        decode_args = {}
        if self.tokenizer.chat_template is not None:
            decode_args["skip_special_tokens"] = True

        for seq in sequences:
            if self.model_type == "CausalLM":
                texts.append(self.tokenizer.decode(seq[input_len:], **decode_args))
            else:
                texts.append(self.tokenizer.decode(seq[1:], **decode_args))

        return texts

    def __call__(self, **args):
        """
        Calls the model on the input batch. Returns the resulted scores.
        """
        return self.model(**args)

    def device(self):
        """
        Returns the device the model is currently loaded on.

        Returns:
            str: device string.
        """
        return self.model.device

    @staticmethod
    def from_pretrained(
        model_path: str,
        generation_params: Optional[Dict] = {},
        add_bos_token: bool = True,
        **kwargs,
    ):
        """
        Initializes the model from HuggingFace. Automatically determines model type.

        Parameters:
            model_path (str): model path in HuggingFace.
            generation_params (Dict): generation arguments for
                lm_polygraph.utils.generation_parametersGenerationParameters
            add_bos_token (bool): tokenizer argument. Default: True.
        """
        log.warning(
            "WhiteboxModel#from_pretrained is deprecated and will be removed in the next release. Please instantiate WhiteboxModel directly by passing an already loaded model, tokenizer and model path."
        )

        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        )

        if any(["CausalLM" in architecture for architecture in config.architectures]):
            model_type = "CausalLM"
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, **kwargs
            )
        elif any(
            [
                ("Seq2SeqLM" in architecture)
                or ("ConditionalGeneration" in architecture)
                for architecture in config.architectures
            ]
        ):
            model_type = "Seq2SeqLM"
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **kwargs)
            if "falcon" in model_path:
                model.transformer.alibi = True
        elif any(
            ["JAISLMHeadModel" in architecture for architecture in config.architectures]
        ):
            model_type = "CausalLM"
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                **kwargs,
            )
        elif any(
            ["BartModel" in architecture for architecture in config.architectures]
        ):
            model_type = "Seq2SeqLM"
            model = BartForConditionalGeneration.from_pretrained(model_path, **kwargs)
        else:
            raise ValueError(
                f"Model {model_path} is not adapted for the sequence generation task"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            add_bos_token=add_bos_token,
            **kwargs,
        )

        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        generation_params = GenerationParametersFactory.from_params(
            yaml_config=generation_params,
            native_config=asdict(model.config),
        )

        instance = WhiteboxModel(
            model, tokenizer, model_path, model_type, generation_params
        )

        return instance

    def tokenize(
        self, texts: Union[List[str], List[List[Dict[str, str]]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenizes input texts batch into a dictionary using the model tokenizer.

        Parameters:
            texts (List[str]): list of input texts batch.
        Returns:
            dict[str, torch.Tensor]: tensors dictionary obtained by tokenizing input texts batch.
        """
        # Apply chat template if tokenizer has it
        add_start_symbol = True
        if self.instruct:
            formatted_texts = []
            for chat in texts:
                if isinstance(chat, str):
                    chat = [{"role": "user", "content": chat}]
                formatted_chat = self.tokenizer.apply_chat_template(
                    chat, add_generation_prompt=True, tokenize=False
                )
                formatted_texts.append(formatted_chat)
            texts = formatted_texts

            add_start_symbol = False
        return self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=add_start_symbol,
        )


def create_ensemble(
    models: List[WhiteboxModel] = [],
    mc: bool = False,
    seed: int = 1,
    mc_seeds: List[int] = [1],
    ensembling_mode: str = "pe",
    dropout_rate: float = 0.1,
    **kwargs,
) -> WhiteboxModel:
    model = models[0]
    ens = model.model

    ens.__class__ = type(
        "EnsembleModel", (model.model.__class__, EnsembleGenerationMixin), {}
    )

    if mc:
        ens.mc = True
        ens.mc_seeds = mc_seeds
        ens.base_seed = seed
        ens.ensembling_mode = ensembling_mode
        ens.mc_models_num = len(mc_seeds)
        ens.mc_seeds = mc_seeds

        replace_dropout(
            ens.config._name_or_path, ens, p=dropout_rate, share_across_tokens=True
        )
        ens.train()
    else:
        raise ValueError(
            "Only Monte-Carlo ensembling is available. Please set the corresponding argument value to True"
        )

    return model
