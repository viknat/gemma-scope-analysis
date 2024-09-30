import streamlit as st
import pandas as pd
import plotly.express as px


def create_neuronpedia_link(feature_index):
    base_url = "https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/"
    return f"{base_url}{feature_index}"


df = pd.read_csv('feature_analysis.csv')

df_grouped = df.groupby(['phrase', 'sub_component'], as_index=False).agg({'total_activation': 'mean', 'feature_index': 'first'})

st.title("Gemma Scope SAE Feature Analysis")

st.write("""
         This is an interactive visualization to explore the atomicity of Gemma Scope SAE features by measuring the 
         extent to which they are activated by conceptual subcomponents.
         Features are sampled from layer 20 of the Gemma 2 2B release.
         """)

selected_phrase = st.selectbox("Select an SAE feature", df_grouped['phrase'].unique())

filtered_df = df_grouped[df_grouped['phrase'] == selected_phrase]

feature_index = filtered_df['feature_index'].unique()[0]
neuronpedia_link = create_neuronpedia_link(feature_index)
st.markdown(f"[View Neuronpedia Dashboard for this feature]({neuronpedia_link})")

if filtered_df['total_activation'].sum() == 0:
    st.write(f"None of the subcomponents for '{selected_phrase}' were activated.")
else:
    fig = px.bar(
        filtered_df,
        x='sub_component',
        y='total_activation',
        title=f"Mean Activation by sub-component for this feature",
        labels={'total_activation': 'Mean Activation', 'sub_component': 'Subcomponent'},
        height=600
    )

    st.plotly_chart(fig)

st.write("Select a subcomponent to view the activating phrases that were used.")

clicked_sub_component = st.selectbox("Select a Subcomponent", filtered_df['sub_component'].unique())

activating_df = df[(df['phrase'] == selected_phrase) & (df['sub_component'] == clicked_sub_component)]

if not activating_df.empty:
    fig2 = px.bar(
        activating_df,
        x='total_activation',
        y='activating_phrase',
        title=f"Activating Phrases for {clicked_sub_component}",
        labels={'total_activation': 'Activation', 'activating_phrase': 'Activating Phrase'},
        height=400
    )
    st.plotly_chart(fig2)
else:
    st.write("No activating phrases found for this subcomponent.")
