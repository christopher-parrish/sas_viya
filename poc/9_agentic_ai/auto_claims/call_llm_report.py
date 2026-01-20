
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

    def set_prompt(self, system_prompt=None, context=None):
        if system_prompt is None:
            system_prompt = "Say 'Hello World'"
        if context is None:
            context = ""
        self.prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{context}"}
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
        print(str(self.get_prompt()))

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
                "driver_at_fault": "ERROR",
                "vehicles_involved": "ERROR",
                "damage_to_each_vehicle": "ERROR"
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

# def execute (report,endpoint,api_key,version,system_prompt_report,deployment,temperature,max_tokens,top_p,frequency_penalty,presence_penalty):
def viya (report,endpoint,api_key,version,system_prompt_report,deployment,temperature,max_tokens,top_p,frequency_penalty,presence_penalty):
    'Output:result'
    available_deployments = {
        'gpt-4': 'chat',
        'gpt-4o': 'chat',
        'gpt-4o-mini': 'chat',
        'gpt-35-turbo': 'chat',
        'o3-mini': 'reasoning',
        'o1': 'reasoning'
    }

    input_data = pd.DataFrame({"DocumentText": [report]})
    text_col = "DocumentText"

    if available_deployments.get(deployment) == 'chat':
        output_df = runLLM(
            azure_openai_endpoint=endpoint,
            azure_key=api_key,
            azure_openai_version=version,
            system_prompt=system_prompt_report,
            input_data=input_data,
            deployment_name=deployment,
            text_col=text_col,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        result = output_df.iloc[0, -1]
    else:
        result = "ERROR FOUND"
    return result


result_image = """The vehicle in the image is a silver 2007-2012 Acura MDX. The identifiable damage includes a significant dent and a tear on the rear bumper, along with scratches and paint damage. The rear left area shows signs of impact, indicating a collision."""
system_prompt_report = f"""You are an AI assistant that helps insurance companies determine the specific type of damage that occurred in an automobile accident. You receive ONE accident report text blob per request. Your job: summarize the vehicles involved in the incident and the damage to each vehicle. Using the following vehicle information, determine whether a vehicle matching or nearly matching the description is involved in the automobile accident report and if so, whether the damage described in the vehicle description matches or nearly matches the damage of the vehicle in the accident report. Vehicle information: "{result_image}". OUTPUT CONTRACT Return valid text. END SYSTEM PROMPT """
report = """On September 8, 2024, at approximately 5:47 PM, I, Lieutenant Katherine M. Hayes (Badge #4512), along with Trooper Michael J. Foster (Badge #7293), responded to a five-vehicle chain-reaction collision on Interstate 40 Eastbound at Mile Marker 287.3. The accident occurred during severe weather conditions with heavy rainfall and dense fog reducing visibility to approximately one-quarter mile. Vehicle 1 (Initial Striking Vehicle): A black 2020 Chevrolet Tahoe (NC License Plate: GBX-5829, VIN: 1GNSKCKC5LR158743) operated by Christopher Alan Brennan (DOB: 04/22/1976, DL #: NC-948372516, Address: 3847 Stonegate Drive, Durham, NC 27707). Mr. Brennan was accompanied by his wife, Patricia Lynn Brennan (DOB: 07/15/1978, Address: same as driver), and their two children: Tyler James Brennan, age 12 (DOB: 11/30/2011) and Madison Grace Brennan, age 9 (DOB: 03/17/2015). All occupants sustained injuries ranging from minor to moderate. Vehicle 2: A silver 2017 Toyota Camry (NC License Plate: FMR-7164, VIN: 4T1BF1FK8HU392847) operated by Dr. Rajesh Kumar Patel (DOB: 09/08/1965, DL #: NC-627394851, Address: 7821 Briarcliff Court, Raleigh, NC 27615). Dr. Patel was alone in the vehicle and sustained moderate injuries including a broken collarbone. Vehicle 3: A white 2019 Honda CR-V (NC License Plate: KJP-9283, VIN: 7FARW2H58KE147529) operated by Elizabeth Marie Thompson (DOB: 01/28/1990, DL #: NC-516847392, Address: 2947 Meadowbrook Lane, Cary, NC 27518). Ms. Thompson's passenger was her elderly mother, Dorothy Ann Thompson (DOB: 05/14/1948, Address: 5628 Sunset Ridge Road, Apex, NC 27502). Both sustained minor injuries. Vehicle 4: A red 2022 Nissan Altima (NC License Plate: NPT-4627, VIN: 1N4BL4BV8NC184759) operated by Kevin Michael O'Connor (DOB: 12/07/1982, DL #: NC-739184625, Address: 8473 Willow Creek Boulevard, Holly Springs, NC 27540). Mr. O'Connor was traveling alone and sustained serious injuries including multiple rib fractures and internal bleeding. Vehicle 5: A blue 2018 Ford Explorer (NC License Plate: HRT-8394, VIN: 1FM5K8D87JGA58291) operated by Maria Isabel Santos (DOB: 08/30/1995, DL #: NC-847293716, Address: 4521 Pine Valley Drive, Raleigh, NC 27603). Ms. Santos was accompanied by her boyfriend, David Anthony Miller (DOB: 06/19/1993, Address: 1847 Oakwood Street, Raleigh, NC 27604). Both sustained minor injuries. The collision sequence began when Mr. Brennan, traveling at approximately 70 mph in heavy rain and fog conditions, failed to observe stopped or slow-moving traffic ahead due to a separate minor accident that had occurred 10 minutes earlier. Despite electronic message boards 2 miles prior warning of the hazard and reduced speed limits, Mr. Brennan did not reduce his speed appropriately for conditions. His Tahoe struck the rear of Dr. Patel's Camry, which was traveling at approximately 25 mph due to the hazardous conditions. The impact caused Dr. Patel's Camry to be propelled forward into the Honda CR-V operated by Ms. Thompson, which was traveling at approximately 30 mph. The Thompson vehicle was then pushed into Mr. O'Connor's Nissan Altima, which had been traveling at 35 mph and attempting to change lanes to avoid the developing situation. Finally, the Altima collided with Ms. Santos's Ford Explorer, which was in the left lane traveling at approximately 40 mph. Multiple witnesses confirmed that Mr. Brennan was following too closely and traveling too fast for the weather conditions. Witness Carl Robert Jenkins (Address: 5847 Forest Hills Drive, Raleigh, NC), driving in the adjacent lane, stated he observed the Tahoe approaching at high speed and not slowing down at all" despite the clearly hazardous conditions and the line of slower-moving vehicles ahead. Emergency Medical Services from Wake County arrived at 6:02 PM, with additional units from Johnston County arriving at 6:08 PM due to the multiple injuries. Life Flight helicopter was dispatched for Mr. O'Connor due to the severity of his injuries but was unable to land due to weather conditions. He was transported by ambulance to WakeMed Raleigh with serious but non-life-threatening injuries. Investigation revealed that Mr. Brennan had been cited twice in the past 18 months for following too closely and once for speeding. His vehicle showed evidence of recent cell phone use, though he denied using the device while driving. Alcohol screening was negative for all drivers involved. Mr. Christopher Alan Brennan is determined to be 100% at fault for initiating this chain-reaction collision due to: Failure to reduce speed for hazardous weather conditions Following too closely for conditions (estimated 1.5 seconds behind lead vehicle) Failure to maintain proper lookout Driving at excessive speed (70 mph) given weather conditions and traffic patterns Disregarding electronic warning signs about hazardous conditions ahead"""
e = 'https://openai-scr.openai.azure.com/'
k = 'XXX'
v = '2025-01-01-preview'
d = 'gpt-4o-mini'
m = 800
te = 0.3
to = 0.95
f = 0.0
p = 0.0

viya(report,e,k,v,system_prompt_report,d,te,m,to,f,p)

prompt_test = system_prompt_report
prompt_test