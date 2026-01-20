
prediction='Real'
result_report = {
  "driver_at_fault": ["Christopher Alan Brennan"],
  "vehicles_involved": ["2020 Chevrolet Tahoe", "2017 Toyota Camry", "2019 Honda CR-V", "2022 Nissan Altima", "2018 Ford Explorer"],
  "damage_to_each_vehicle": [
    "Front damage from collision with Camry",
    "Rear damage from being pushed into CR-V",
    "Front damage from being pushed into Altima",
    "Rear damage from collision with Explorer",
    "Rear damage from being hit by Altima"
  ]
}
result_image = {
  "make": "Acura",
  "model": ["MDX"],
  "color": ["silver"],
  "damage_to_vehicle": [
    "Rear bumper damage",
    "Scratches on rear panel",
    "Possible dent on rear left side"
  ]
}

def sysprompt(prediction, result_report, result_image):
    "Output: agent_summary"
        
    agent_summary = "The submitted image appears to be a silver Acura MDX with rear bumper damage, scratches on the rear panel and a possible/n" \
    " dent on the rear left side.  Based on computer vision model, the submitted photo is categorized as f'{prediction}'. n/" \
    "The submitted auto damage report indicates that there were five vehicles involved, including ...  "

    return agent_summary

sysprompt(prediction, result_report, result_image)