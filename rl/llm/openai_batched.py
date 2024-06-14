import io
import json
from datetime import datetime
from pathlib import Path

import openai

from rl.llm.engines import ChatInput, InferenceOutput
from rl.utils.io import get_data_path, getenv

OPENAI_API_KEY = getenv("OPENAI_API_KEY")
OPENAI_ORGANIZATION = getenv("OPENAI_ORGANIZATION")

class OpenAIBatch:
    model: str
    request: list[list[ChatInput]]
    client: openai.OpenAI
    file_id: str | None = None
    batch_id: str | None = None
    response: list[InferenceOutput] | None = None
    max_tokens: int = 1000

    def __init__(
        self,
        request: list[list[ChatInput]],
        model: str,
        max_tokens: int = 1000,
        file_id: str | None = None,
        batch_id: str | None = None,
        response: list[InferenceOutput] | None = None,
        id_prefix: str = "batch-inference-",
    ) -> None:
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORGANIZATION)
        self.request = request
        self.model = model
        self.max_tokens = max_tokens
        self.id_prefix = id_prefix
        self.file_id = file_id
        self.batch_id = batch_id
        self.response = response


    def prepare_batch(self) -> list[dict]:
        batch_template = {
            "custom_id": None,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": None,
                "max_tokens": self.max_tokens,
            },
        }
        batch = []
        for n, req in enumerate(self.request):
            formatted_request = dict(batch_template)
            formatted_request["custom_id"] = f"{self.id_prefix}{n}"
            formatted_request["body"]["messages"] = req
            batch.append(formatted_request)
        return batch


    def upload_file(self) -> str:
        jsonl_content = "\n".join([json.dumps(b) for b in self.prepare_batch()])
        json_byte_stream = io.BytesIO(jsonl_content.encode("utf-8"))
        batch_input_file = self.client.files.create(
            file=json_byte_stream,
            purpose="batch",
        )
        self.file_id = batch_input_file.id
        return self.file_id


    def create_batch(self, metadata={}) -> str:
        if not self.file_id:
            self.upload_file()
        batch = self.client.batches.create(
            input_file_id=self.file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata=metadata,
        )
        self.batch_id = batch.id
        return self.batch_id


    def check_status(self) -> str:
        if not self.batch_id:
            raise ValueError("No batch ID found. Please create a batch first.")
        return self.client.batches.retrieve(self.batch_id).status


    def get_response(self) -> list[InferenceOutput]:
        if not self.file_id or not self.batch_id:
            raise ValueError("No batch ID found. Please create a batch first.")
        response_file_id = self.client.batches.retrieve(self.batch_id).output_file_id
        content = self.client.files.content(file_id=response_file_id)
        response = [json.loads(line) for line in content.read().decode("utf-8").split("\n") if line]
        request_map = {
            r["custom_id"]: r["body"]["messages"]
            for r in self.prepare_batch()
        }
        response_map = {
            r["custom_id"]: r
            for r in response
        }
        inference_outputs = [
            InferenceOutput(
                prompt=req,
                text=response_map[req["custom_id"]]["response"]["body"]["choices"][0]["message"]["content"],
                metadata=response_map[req["custom_id"]]
            )
            for req in self.prepare_batch()
        ]
        self.response = inference_outputs
        return inference_outputs


    def serialized_response(self) -> list[dict] | None:
        if not self.response:
            return None
        return [
            {
                "text": r.text,
                "prompt": r.prompt,
                "metadata": r.metadata,
            }
            for r in self.response
        ]

    @classmethod
    def deserialize_response(cls, response: list[dict]) -> list[InferenceOutput]:
        if not response:
            return None
        return [
            InferenceOutput(
                prompt=r["prompt"],
                text=r["text"],
                metadata=r["metadata"],
            )
            for r in response
        ]


    def write(self, path: Path | str | None=None) -> None:
        if not path:
            path = get_data_path() / f"openai_batch_{datetime.now().isoformat()}.json"
        if isinstance(path, str):
            path = Path(path)
        write_dict = {
            "model": self.model,
            "request": self.request,
            "file_id": self.file_id,
            "batch_id": self.batch_id,
            "response": self.serialized_response(),
            "max_tokens": self.max_tokens,
            "id_prefix": self.id_prefix,
        }
        with open(path, "w") as f:
            json.dump(write_dict, f)


    @classmethod
    def read(self, path: Path | str) -> "OpenAIBatch":
        if isinstance(path, str):
            path = Path(path)
        with open(path, "r") as f:
            read_dict = json.load(f)
        return OpenAIBatch(
            request=read_dict["request"],
            model=read_dict["model"],
            file_id=read_dict["file_id"],
            batch_id=read_dict["batch_id"],
            response=OpenAIBatch.deserialize_response(read_dict["response"]),
            max_tokens=read_dict["max_tokens"],
            id_prefix=read_dict.get("id_prefix", "batch-inference-"),
        )
