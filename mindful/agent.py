import json
import os
import openai
import requests
import time


class Agent:
    model_providers = {
        "anthropic": "ANTHROPIC_API_KEY",
        "azure_openai": "AZURE_API_KEY",
        "cohere": "COHERE_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "ollama": "OLLAMA_API_KEY",
        "huggingface": "HUGGINGFACE_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "replicate": "REPLICATE_API_KEY",
        "together": "TOGETHERAI_API_KEY",
        "xai": "XAI_API_KEY",
    }

    def __init__(self, model: str):
        if model not in Agent.model_providers:
            raise ValueError(f"Unsupported model: {model}")

        self.model = model
        self._api_key = os.environ.get(Agent.model_providers[model])

        if not self._api_key:
            raise ValueError(f"API key for {model} not found in environment variables.")

    def _generate_with_openai(self, prompt: str) -> str:
        """Use OpenAI to generate content."""
        openai.api_key = self._api_key
        response = openai.Completion.create(
            engine="text-embedding-3-large", prompt=prompt, max_tokens=150, n=1, stop=None, temperature=0.7
        )
        return response.choices[0].text.strip()

    def _generate_with_anthropic(self, prompt: str) -> str:
        """Use Anthropic to generate content."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        response = requests.post(
            "https://api.anthropic.com/v1/completions", headers=headers, json={"prompt": prompt, "max_tokens": 150}
        )
        response.raise_for_status()
        return response.json().get("text", "").strip()

    def _generate_with_cohere(self, prompt: str) -> str:
        """Use Cohere to generate content."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        response = requests.post(
            "https://api.cohere.ai/generate", headers=headers, json={"prompt": prompt, "max_tokens": 150}
        )
        response.raise_for_status()
        return response.json().get("text", "").strip()

    def _generate_with_azure_openai(self, prompt: str) -> str:
        """Use Azure OpenAI to generate content."""
        if "AZURE_API_BASE" not in os.environ:
            raise ValueError("AZURE_API_BASE environment variable must be set.")

        original_api_base = openai._client._base
        original_api_key = openai._client._key

        try:
            openai._client._base = os.environ["AZURE_API_BASE"]
            openai._client._key = self._api_key

            response = openai.Completion.create(
                engine="text-davinci-003", prompt=prompt, max_tokens=150, n=1, stop=None, temperature=0.7
            )
            return response.choices[0].text.strip()

        finally:
            openai._client._base = original_api_base
            openai._client._key = original_api_key

    def _generate_with_deepseek(self, prompt: str) -> str:
        """Use DeepSeek to generate content."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        response = requests.post(
            "https://api.deepseek.com/v1/completions", headers=headers, json={"prompt": prompt, "max_tokens": 150}
        )
        response.raise_for_status()
        return response.json().get("text", "").strip()

    def _generate_with_ollama(self, prompt: str) -> str:
        """Use Ollama to generate content."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        response = requests.post(
            "https://api.ollama.ai/v1/completions", headers=headers, json={"prompt": prompt, "max_tokens": 150}
        )
        response.raise_for_status()
        return response.json().get("text", "").strip()

    def _generate_with_huggingface(self, prompt: str) -> str:
        """Use Hugging Face Inference API to generate content."""
        model_name = "EleutherAI/gpt-neo-2.7B"
        url = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        data = {"inputs": prompt}
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()[0]["generated_text"].strip()

    def _generate_with_gemini(self, prompt: str) -> str:
        """Use Google Gemini API to generate content."""
        url = "https://generativelangaugeapis.google.com/v1beta2/models/gemini-pro:generateContent"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        data = {"prompt": prompt}
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json().get("text", "").strip()

    def _generate_with_groq(self, prompt: str) -> str:
        """Use Groq AI API to generate content."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        response = requests.post(
            "https://api.groq.com/v1/completions", headers=headers, json={"prompt": prompt, "max_tokens": 150}
        )
        response.raise_for_status()
        return response.json().get("text", "").strip()

    def _generate_with_mistral(self, prompt: str) -> str:
        """Use Mistral AI API to generate content."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        response = requests.post(
            "https://api.mistral.ai/v1/completions", headers=headers, json={"prompt": prompt, "max_tokens": 150}
        )
        response.raise_for_status()
        return response.json().get("text", "").strip()

    def _generate_with_replicate(self, prompt: str) -> str:
        """Use Replicate AI API to generate content."""
        model_id = "some-model-id-for-text-generation"
        url = "https://api.replicate.com/v1/predictions"
        headers = {
            "Authorization": f"Token {self._api_key}",
        }
        data = {"model": model_id, "input": {"prompt": prompt}}
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        prediction_id = response.json()["id"]
        start_time = time.time()
        while time.time() - start_time < 60:
            status_response = requests.get(f"{url}/{prediction_id}", headers=headers)
            status_response.raise_for_status()
            status = status_response.json()["status"]
            if status == "succeeded":
                output_response = requests.get(f"{url}/{prediction_id}/output", headers=headers)
                output_response.raise_for_status()
                return output_response.json().get("text", "").strip()
            elif status == "failed":
                raise Exception("Prediction failed")
            time.sleep(1)
        raise TimeoutError("Prediction did not complete within 60 seconds")

    def _generate_with_together(self, prompt: str) -> str:
        """Use Together AI API to generate content."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        response = requests.post(
            "https://api.together.ai/v1/completions", headers=headers, json={"prompt": prompt, "max_tokens": 150}
        )
        response.raise_for_status()
        return response.json().get("text", "").strip()

    def _generate_with_xai(self, prompt: str) -> str:
        """Use xAI API to generate content."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        response = requests.post(
            "https://api.x.ai/v1/completions", headers=headers, json={"prompt": prompt, "max_tokens": 150}
        )
        response.raise_for_status()
        return response.json().get("text", "").strip()

    def generate_content(self, prompt: str) -> str:
        """Generates content using the selected model."""
        if self.model == "openai":
            return self._generate_with_openai(prompt)
        elif self.model == "anthropic":
            return self._generate_with_anthropic(prompt)
        elif self.model == "cohere":
            return self._generate_with_cohere(prompt)
        elif self.model == "azure_openai":
            return self._generate_with_azure_openai(prompt)
        elif self.model == "deepseek":
            return self._generate_with_deepseek(prompt)
        elif self.model == "ollama":
            return self._generate_with_ollama(prompt)
        elif self.model == "huggingface":
            return self._generate_with_huggingface(prompt)
        elif self.model == "gemini":
            return self._generate_with_gemini(prompt)
        elif self.model == "groq":
            return self._generate_with_groq(prompt)
        elif self.model == "mistral":
            return self._generate_with_mistral(prompt)
        elif self.model == "replicate":
            return self._generate_with_replicate(prompt)
        elif self.model == "together":
            return self._generate_with_together(prompt)
        elif self.model == "xai":
            return self._generate_with_xai(prompt)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def generate_metadata(self, content: str) -> dict:
        """Generate dynamic metadata based on the given content using the selected model."""
        prompt = f"""
            You are a metadata extraction assistant. Given the content below, generate metadata in JSON format with the following fields: "category" (broad classification), "context" (situational or thematic context), and "keywords" (list of key terms). Keep it concise, accurate, and safe—no harmful or off-topic content.

            Content: {content}

            Output: ```json
            {{
            "category": "",
            "context": "",
            "keywords": []
            }}
            ```
        """
        response = self.agent.generate_content(prompt)

        try:
            # Extract the JSON part from the response
            json_str = response.split("```json")[1].split("```")[0].strip()
            metadata = json.loads(json_str)
            return metadata["category"], metadata["context"], metadata["keywords"]
        except (IndexError, ValueError) as e:
            print(f"Error extracting metadata: {e}")
            return "unknown", "unknown", []  # Default return values
