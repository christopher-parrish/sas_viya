
import os, json, time, requests
import pandas as pd
import numpy as np

# function to initalize Open AI model with preset parameters as rules
# parameters are defined in rules node atop the decision and mapped in last function within this script
class SASAzureOpenAILLM():
    def __init__(self, client=None, azure_openai_endpoint=None, deployment_name=None, azure_key=None,
                 azure_openai_version=None, temperature=None, max_tokens=None, top_p=None,
                 frequency_penalty=None, presence_penalty=None):
        self.client = None  # not used with requests; kept for API compatibility
        self.azure_openai_endpoint = azure_openai_endpoint
        self.deployment_name = deployment_name
        self.azure_key = azure_key
        self.azure_openai_version = azure_openai_version
        self.prompt = []
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    def set_client(self, azure_openai_endpoint=None, azure_key=None, azure_openai_version=None):
        # Endpoint
        if azure_openai_endpoint is None:
            try:
                azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
            except KeyError:
                raise ValueError("Endpoint must be provided or set in AZURE_OPENAI_ENDPOINT")

        # Key (support either name)
        if azure_key is None:
            azure_key = os.environ.get("AZURE_OPENAI_AZURE_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")
            if not azure_key:
                raise ValueError("API key must be provided or set in AZURE_OPENAI_AZURE_KEY or AZURE_OPENAI_API_KEY")

        # API version
        if azure_openai_version is None:
            try:
                azure_openai_version = os.environ["AZURE_OPENAI_API_VERSION"]
            except KeyError:
                raise ValueError("API version must be provided or set in AZURE_OPENAI_API_VERSION")

        self.azure_openai_endpoint = azure_openai_endpoint.rstrip("/")
        self.azure_key = azure_key
        self.azure_openai_version = azure_openai_version

    def get_client(self):
        # Kept for compatibility with your code
        if not (self.azure_openai_endpoint and self.azure_key and self.azure_openai_version):
            raise ValueError("Client not set. Call set_client() first.")
        return {"endpoint": self.azure_openai_endpoint, "key": self.azure_key, "version": self.azure_openai_version}

# does context pull in the user prompt here?
    def set_prompt(self, system_prompt=None, context=None):
        if system_prompt is None:
            system_prompt = "Say 'Hello World'"
        if context is None:
            context = ""
        print(context)
        self.prompt = [
            {"role": "system", "content": system_prompt},
            #{"role": "user", "content": f"{context}"} for pure text user prompt
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{context}"}}]}
        ]

    def get_prompt(self):
        return " ".join((self.prompt[0]["content"], self.prompt[1]["content"]))

    # --- internal helper using requests ---
    def _post_chat_completion(self, deployment_name, messages, temperature, max_tokens, top_p, frequency_penalty, presence_penalty,
                              timeout=30, max_retries=3):
        url = f"{self.azure_openai_endpoint}/openai/deployments/{deployment_name}/chat/completions"
        params = {"api-version": self.azure_openai_version}
        headers = {
            "Content-Type": "application/json",
            "api-key": self.azure_key,
        }
        body = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        # prune None values so you inherit deployment defaults where not set
        body = {k: v for k, v in body.items() if v is not None}

        # simple retry loop for 429/5xx
        backoff = 1.0
        for attempt in range(1, max_retries + 1):
            resp = requests.post(url, headers=headers, params=params, json=body, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
                continue
            # raise on final failure
            raise RuntimeError(f"Azure OpenAI error {resp.status_code}: {resp.text}")

# does context pull in the user prompt here?
    def get_response(self, context=None, client=None, deployment_name=None, prompt=None,
                     system_prompt=None, temperature=None, max_tokens=None, top_p=None,
                     frequency_penalty=None, presence_penalty=None):
        # Return blank if no context
        if isinstance(context, float) and np.isnan(context):
            print("No context provided"); return ""
        if context is None or len(str(context)) == 0:
            print("No context provided"); return ""

        # Build messages
        self.set_prompt(system_prompt, context)
        print("Context Found: ")
        # print(str(self.get_prompt()))

        # Resolve parameters
        _ = self.get_client()  # ensures set_client was called
        deployment_name = deployment_name or self.deployment_name
        messages = prompt or self.prompt
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        top_p = top_p if top_p is not None else self.top_p
        frequency_penalty = frequency_penalty if frequency_penalty is not None else self.frequency_penalty
        presence_penalty = presence_penalty if presence_penalty is not None else self.presence_penalty

        try:
            data = self._post_chat_completion(
                deployment_name=deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            # mirror SDK accessor
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print("Error calling Azure OpenAI API (requests):", e)
            return json.dumps({
                "make": "ERROR",
                "model": "ERROR",
                "color": "ERROR",
                "damage_to_vehicle": "ERROR"
                #"version": "v3.2",
            })

def runLLM(azure_openai_endpoint=None, azure_key=None, azure_openai_version=None, system_prompt=None,
           input_data=None, deployment_name=None, text_col=None,
           temperature=None, max_tokens=None, top_p=None, frequency_penalty=None, presence_penalty=None):

    model = SASAzureOpenAILLM(temperature=temperature, max_tokens=max_tokens, top_p=top_p,
                              frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
    model.set_client(azure_openai_endpoint, azure_key, azure_openai_version)
    # apply returns a Series with responses
    input_data['Response'] = input_data[text_col].apply(
        model.get_response,
        deployment_name=deployment_name,
        system_prompt=system_prompt
    )
    return input_data

def viya (image,endpoint,api_key,version,system_prompt_image,deployment,temperature,max_tokens,top_p,frequency_penalty,presence_penalty):
    'Output:result'
    available_deployments = {
        'gpt-4': 'chat',
        'gpt-4o': 'chat',
        'gpt-4o-mini': 'chat',
        'gpt-35-turbo': 'chat',
        'o3-mini': 'reasoning',
        'o1': 'reasoning'
    }

    input_data = pd.DataFrame({"DocumentText": [image]})
    text_col = "DocumentText"

    if available_deployments.get(deployment) == 'chat':
        output_df = runLLM(
            azure_openai_endpoint=endpoint,
            azure_key=api_key,
            azure_openai_version=version,
            system_prompt=system_prompt_image,
            input_data=input_data,
            deployment_name=deployment,
            text_col=text_col,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        result_image = output_df.iloc[0, -1]
        print(result_image)
    else:
        result_image = "ERROR FOUND"
    return result_image


import base64
system_prompt_img = """You are an AI assistant that helps insurance companies assess vehicle damage. You receive ONE base64 encoded image text blob per request. Your job: identify the make, model, color and any identifiable damage related to the vehicle damaged in the attached image and summarize the results succinctly as this summarization will be appended to an accident report. OUTPUT CONTRACT Return valid text. END SYSTEM PROMPT """
image_path = ".../images/auto_damage/real_damage/IMG_1520.JPG"
encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii') #decode.('utf-8)
image_prompt = f"data:image/jpeg;base64,{encoded_image}"
e = 'https://openai-scr.openai.azure.com/'
k = 'XXX'
v = '2025-01-01-preview'
d = 'gpt-4o-mini'
m = 800
te = 0.3
to = 0.95
f = 0.0
p = 0.0

viya(encoded_image,e,k,v,system_prompt_img,d,te,m,to,f,p)
