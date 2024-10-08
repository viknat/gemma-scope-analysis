#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install sae_lens transformer_lens openai')
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from huggingface_hub import hf_hub_download, login
import numpy as np
import torch
from sae_lens import SAE, HookedSAETransformer
import plotly.express as px
from google.colab import userdata
import requests
import json
import plotly.express as px
from google.colab import userdata
from openai import OpenAI


# In[ ]:


login(token=userdata.get("HF_API_KEY"))


# In[ ]:


sae, cfg, sparsity = SAE.from_pretrained(
    release = "gemma-scope-2b-pt-res",
    sae_id = "layer_20/width_16k/average_l0_71",
    device = "cuda"
)


# In[ ]:


sae.cfg.d_sae


# In[ ]:


from IPython.display import IFrame

feature_idx = torch.randint(0, sae.cfg.d_sae, (1,)).item()

html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"

def get_dashboard_html(sae_release, sae_id, feature_idx=0):
    return html_template.format(sae_release, sae_id, feature_idx)

html = get_dashboard_html(sae_release = "gemma-2-2b", sae_id="20-gemmascope-res-16k", feature_idx=feature_idx)
IFrame(html, width=1200, height=600)


# In[ ]:


def get_neuronpedia_feature(model_id, sae_id, feature_index):
    url = f'https://www.neuronpedia.org/api/feature/{model_id}/{sae_id}/{feature_index}'

    headers = {
        # "Content-Type": "application/json",
        "X-Api-Key": userdata.get('NEURONPEDIA_API_KEY')
    }

    # payload = {
    #     "modelId": model_id,
    #     "saeId": sae_id
    # }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code}, {response.text}"

def get_neuronpedia_custom_text_activation(model_id, layer, feature_index, custom_text):
    url = f'https://www.neuronpedia.org/api/activation/new'
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": userdata.get('NEURONPEDIA_API_KEY')
    }
    data = {
      "feature": {
        "modelId": model_id,
        "layer": layer,
        "index": feature_index,
      },
      "customText": custom_text
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code}, {response.text}"


# In[ ]:


client = OpenAI(
    api_key=userdata.get('OPENAI_API_KEY')
)


# In[ ]:


prompt_template = """Given a particular phrase, which represents an sparse autoencoder feature, please come up with a list of five words, which represents sub-components of that feature, or bits of meaning that individually comprise that feature. Additionally for each word generated in this way, generate three sentences or phrases that you think would "activate" this feature in a deep neural network.

For example, for Einstein, you should return words like "physicist", "German", "Relativity" and "genius". For "physicist", a sentence or phrase could be "he went into the lab to conduct experiments on optics, electromagnetism and gravity to discover the laws of the universe."

Here is the phrase: {feature_description}. The sub components and activating phrases should not use words from this phrase

Please return the output in JSON format. Return only the JSON, nothing else. The format should match the following example:
{{
  "phrase": "information related to transportation and logistics, particularly concerning air travel and vehicle access",
  "sub_components": [
    {{
      "word": "air travel",
      "activating_phrases": [
        "The airline implemented new protocols to streamline passenger check-in and boarding procedures.",
        "The flight attendants ensured that all safety measures were followed during turbulence.",
        "Airport lounges were upgraded to offer a more comfortable experience for travelers."
      ]
    }},
    {{
      "word": "vehicle access",
      "activating_phrases": [
        "The new parking system allows for easy access to both short-term and long-term vehicle storage.",
        "Automated gates improved vehicle access to restricted areas in the facility.",
        "The facility's entrance was redesigned to facilitate better vehicle flow and reduce congestion."
      ]
    }},
    {{
      "word": "cargo management",
      "activating_phrases": [
        "Efficient cargo management practices ensured that all freight arrived at its destination without delay.",
        "The cargo handling team used advanced tracking systems to monitor shipments in real-time.",
        "Improved warehousing techniques increased the speed and accuracy of cargo processing."
      ]
    }},
    {{
      "word": "supply chain",
      "activating_phrases": [
        "The supply chain team optimized routes to reduce delays and minimize costs in the distribution process.",
        "Advanced analytics were used to predict demand and adjust supply chain operations accordingly.",
        "Collaboration with local suppliers improved the efficiency of the supply chain network."
      ]
    }},
    {{
      "word": "airport security",
      "activating_phrases": [
        "Enhanced airport security measures were introduced to ensure the safety of all travelers and staff.",
        "Security screenings were expedited through the use of advanced scanning technology.",
        "The security team conducted regular drills to prepare for potential threats and ensure quick responses."
      ]
    }}
  ]
}}
"""


# In[ ]:


all_data = []


# In[ ]:


NUM_ITERATIONS = 50

def generate_feature_index(sae):
    """Generate a random feature index."""
    return np.random.randint(0, sae.cfg.d_sae)

def get_feature_prompt(feature_description, prompt_template):
    """Format the feature prompt using the description."""
    return prompt_template.format(feature_description=feature_description)

def get_chat_response(client, feature_prompt):
    """Get the chat completion response."""
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": feature_prompt}],
        model="gpt-3.5-turbo"
    )
    raw_response = chat_completion.choices[0].message.content
    return raw_response.replace(r'\n', '').replace(r'\t', '')

def parse_chat_response(raw_response):
    """Parse the cleaned chat response into JSON."""
    return json.loads(raw_response)

def attach_neuronpedia_data(sub_components, feature_index):
    """Attach Neuronpedia data to each sub-component's activating phrases."""
    for sub_component in sub_components:
        sub_component['neuronpedia_data'] = [
            get_neuronpedia_custom_text_activation("gemma-2-2b", "20-gemmascope-res-16k", feature_index, phrase)
            for phrase in sub_component['activating_phrases']
        ]
    return sub_components


def process_iteration(sae, client, prompt_template):
    """Process a single iteration of data generation and processing."""
    feature_index = generate_feature_index(sae)
    neuronpedia_feature = get_neuronpedia_feature("gemma-2-2b", "20-gemmascope-res-16k", feature_index)
    feature_prompt = get_feature_prompt(neuronpedia_feature['explanations'][0]['description'], prompt_template)

    raw_response = get_chat_response(client, feature_prompt)
    json_data = parse_chat_response(raw_response)

    json_data['sub_components'] = attach_neuronpedia_data(json_data['sub_components'], feature_index)

    return json_data

def process_all_iterations(sae, client, prompt_template, num_iterations=1):
    """Main function to handle the iterative process."""
    data = []
    for _ in range(num_iterations):
        iteration_data = process_iteration(sae, client, prompt_template)
        data.append(iteration_data)
        all_data.append(iteration_data)
    return data

this_data = process_all_iterations(sae, client, prompt_template, NUM_ITERATIONS)


# In[ ]:


all_data


# In[ ]:


filtered_data = []
# filtered_all_data = [x for x in all_data if any(y != 'Error: 405, ' for y in x['sub_components']['neuronpedia_data'])]
for i, data in enumerate(all_data):
  for sub_component in data["sub_components"]:
    for k in range(len(sub_component['activating_phrases'])):
      data_point = sub_component['neuronpedia_data'][k]
      if data_point == 'Error: 405, ':
        continue
      max_value = data_point['maxValue']
      max_value_token_index = data_point['maxValueTokenIndex']
      max_token = data_point['tokens'][max_value_token_index]
      feature_index = data_point['index']
      num_tokens_activated = sum(1 if x > 0 else 0 for x in data_point['values'])
      total_activation = sum(data_point['values'][1:]) # ignore BOS
      filtered_data.append({
          "iteration": i,
          "phrase": data["phrase"],
          "sub_component": sub_component["word"],
          "activating_phrase": sub_component["activating_phrases"][k],
          "max_value": max_value,
          "max_token": max_token,
          "feature_index": feature_index,
          "num_tokens_activated": num_tokens_activated,
          "total_activation": total_activation,
          "raw_data": data_point
      })


# In[ ]:


import pandas as pd
df = pd.DataFrame(filtered_data)


# In[ ]:


import pandas as pd
import plotly.express as px

df_grouped = df.groupby(['phrase', 'sub_component'], as_index=False).agg({'max_value': 'mean'})

def create_interactive_plot(df_grouped):
    fig = px.bar(
        df_grouped,
        x="sub_component",
        y="max_value",
        color="sub_component",
        facet_col="phrase",
        title="Average Total Activation by Activating Phrase",
        labels={"max_value": "Average Activation", "sub_component": "Component"},
        height=600
    )

    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [{
                            "visible": [phrase == selected_phrase for phrase in df_grouped['phrase'].unique()],
                            "xaxis.showticklabels": [True if phrase == selected_phrase else False for phrase in df_grouped['phrase'].unique()]
                            }],
                        "label": selected_phrase,
                        "method": "update"
                    } for selected_phrase in df_grouped['phrase'].unique()
                ],
                "direction": "down",
                "showactive": True,
            }
        ],
        margin=dict(t=50, b=100),  # Adjust bottom margin
    )

    fig.for_each_trace(lambda trace: trace.update(visible=False) if trace.name != df_grouped['phrase'].unique()[0] else trace.update(visible=True))
    fig.update_xaxes(showticklabels=True, matches=None)

    return fig

fig = create_interactive_plot(df_grouped)

fig.show()


# In[ ]:


from google.colab import files
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"data_{timestamp}.csv"
df.to_csv(filename)
files.download(filename)


# In[ ]:




