import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.io as pio
import AMPnet_functions

# Open FASTA examples file and read contents
file_path = "Resources/fasta_examples.txt"
with open(file_path, "r") as file:
    example_text = file.read()

# Set Display Options to prevent Exponential Notation
pd.set_option('display.float_format', '{:.2f}'.format)

# Function to process the input text
def process_text(user_text):
    try:
        # Process FASTA input from User into Pandas DataFrame
        user_text_df = AMPnet_functions.process_fasta(user_text)

        # Calculate Molecular Weight of Peptides
        user_text_df['MW (kDa)'] = AMPnet_functions.calc_molecular_weight(user_text_df['Sequence'])

        # Calculate Isoelectric Point (pI) of Peptides
        user_text_df['Isoelectric Point (pI)'] = AMPnet_functions.calculate_pI(user_text_df['Sequence'])
        
        # Calculate Hydrophobicity (Kyte-Doolittle Scores) for each Amino Acid
        user_text_df['Hydrophobicity'] = user_text_df['Sequence'].apply(AMPnet_functions.list_hydrophobicities)
        
        # Creae One-Hot Encoded Sequence Column
        user_text_df['One_Hot_Encoded'] = user_text_df['Sequence'].apply(AMPnet_functions.one_hot_encode)
        
        # Set Max Amino Acid Sequence Length
        max_length = 198

        # Pad 'One_Hot_Encoded' Column so that all Matrices are the same size (198 x 20)
        user_text_df['Padded_One_Hot'] = user_text_df['One_Hot_Encoded'].apply(
            lambda arr_list: AMPnet_functions.pad_arrays(arr_list, max_length))

        # Take Values for TensorFlow CNN Model
        user_text_sequences = user_text_df['Padded_One_Hot'].values

        # Convert Input Data to Numpy Array
        user_text_sequences = np.array([np.array(val) for val in user_text_sequences])

        # Load Trained Model
        model = tf.keras.models.load_model('HDF5_files/convolutional_nn_1.h5')

        # Make Predictions
        predictions = model.predict(user_text_sequences)
        predictions = predictions.ravel().tolist()

        # Create DataFrame containing Prediction Results for Plotting
        results_df = pd.DataFrame({'ID': user_text_df['Sequence_ID'],
                                'Sequence': user_text_df['Sequence'],
                                'MW (kDa)': user_text_df['MW (kDa)'],
                                'Isoelectric Point (pI)': user_text_df['Isoelectric Point (pI)'],
                                'AMP Score': predictions}).reset_index(drop=True)

        # Create Classification column based on AMP Score column
        results_df['Classification'] = np.where(results_df['AMP Score'] < 0.5, 'Non-AMP', 'AMP')
        
        return results_df
    
    # Error handling
    except Exception as e:
        st.error("Please ensure that your amino acid sequence(s) are in FASTA format and \
            contain no more than 198 residues per peptide. Please see example below.")
        st.text("Example:")
        st.text(example_text)
        st.stop()

def main():
    # Display AMPnet logo and Title
    col1, col2 = st.columns([0.8,2])
    col1.image("Resources/Images/ampnet_logo.png", width=195)
    col2.title("Antimicrobial Peptide Prediction")
    col2.write("a TensorFlow and Keras deep convolutional neural network that predicts antimicrobial \
               peptides (AMPs) based on amino acid sequence") 
    st.write("---")
    
    # Takes FASTA Input from User
    user_text = st.text_area("Input peptide sequence(s) in FASTA format:")

    # Process User Input
    if st.button("Process"):
        # Create Pandas DataFrame from User Input
        results_df = process_text(user_text)
        st.write("---")
        st.write("#### Results")
        st.dataframe(results_df, hide_index=True)
        
        # Plot Bubble Chart
        bubble_chart = AMPnet_functions.create_bubble_chart(results_df)
        st.plotly_chart(bubble_chart)

    # Link to GitHub
    st.write(" ")
    st.markdown("Created by Samuel Palframan - [GitHub](https://github.com/PianoPalf)")

if __name__ == "__main__":
    main()