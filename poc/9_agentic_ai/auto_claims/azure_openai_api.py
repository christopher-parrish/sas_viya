
import os
import base64
from openai import AzureOpenAI

endpoint = os.getenv("ENDPOINT_URL", "https://openai-scr.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "XXX")

# Initialize Azure OpenAI client with key-based authentication
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2025-01-01-preview",
)

# IMAGE_PATH = "C:/Users/chparr/OneDrive - SAS/git/sas_viya/data/images/auto_damage/real_damage/IMG_1520.JPG"
# encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')
#print (encoded_image)

# Prepare the chat prompt
chat_prompt = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are an AI assistant that helps insurance companies assess vehicle damage. You receive ONE base64 encoded image text blob per request. Output valid JSON ONLY (no backticks, no prose). Your job: identify the make, model, color and any identifiable damage related to the vehicle damaged in the attached image."
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
               "type": "image_url",
                "image_url": {"url": f"https://github.com/christopher-parrish/sas_viya/blob/main/data/images/auto_damage/real_damage/IMG_0143.JPG?raw=true"}
            }
        ]
    }
]

# Include speech result if speech is enabled
messages = chat_prompt

# Generate the completion
completion = client.chat.completions.create(
    model=deployment,
    messages=messages,
    max_tokens=6553,
    temperature=0.7,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None,
    stream=False
)

print(completion.to_json())
    
# "text": "You are an AI assistant that helps insurance companies determine the specific type of damage that occurred in an automobile accident. You receive ONE accident report text blob per request. Output valid JSON ONLY (no backticks, no prose). Your job: summarize the driver at fault and the damage caused to each vehicle involved in the incident."
# "text": "On 2025-07-31 at approximately 10:45 AM EDT, I, Officer J. Smith, responded to a motor vehicle collision at the intersection of Main St and Elm Ave. Upon arrival, two vehicles were involved: a 2018 Honda Civic (Unit 1) and a 2022 Toyota Camry (Unit 2). Driver of Unit 1, Jane Doe, stated she was traveling southbound on Main St and was momentarily distracted by her car radio, causing her to not notice Unit 2 stopped ahead. Unit 1 subsequently impacted the rear of Unit 2. Driver of Unit 2, John Smith, stated he was stopped at the intersection, waiting to make a left turn onto Elm Ave, when his vehicle was struck from behind by Unit 1. A passenger in Unit 1, Emily Doe, complained of minor neck pain but refused EMS transport. Based on driver statements, witness testimony, and physical evidence at the scene (lack of skid marks from Unit 1, rear-end damage to Unit 2, front-end damage to Unit 1), it is determined that Unit 1 failed to reduce speed and was inattentive, leading to the collision. No citations were issued at the scene, but a warning was given to the driver of Unit 1 regarding distracted driving. Unit 1 was towed from the scene due to damage."