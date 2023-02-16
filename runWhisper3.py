import whisper
import gradio as gr 
import warnings
import openai
from gtts import gTTS

warnings.filterwarnings("ignore")
model = whisper.load_model("base")

def transcribe(audio):

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)

    # decode the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    result_text = result.text

    # Call OpenAI API for text for reply back.
    openai.api_key = "<OpenAI API Key>"
    result = openai.Completion.create(
                model="text-davinci-003",
                prompt=result_text,
                max_tokens=500,
                temperature=0
                )  
    out_result = result["choices"][0]["text"]
    speech = gTTS(out_result)
    speech.save("test.mp3")

    return [result_text, out_result, 'test.mp3']


output_1 = gr.Textbox(label="Speech to Text")
output_2 = gr.Textbox(label="ChatGPT Output")
output_3 = gr.Audio(label="ChatGPT Audio Output", value="test.mp3")

gr.Interface(
    title = 'Voice to Text & Voice reply using OpenAI (KF)', 
    fn=transcribe, 
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath")
    ],

    outputs=[
        output_1,  output_2, output_3
    ],
    live=True, allow_flagging=False).launch(share=True)