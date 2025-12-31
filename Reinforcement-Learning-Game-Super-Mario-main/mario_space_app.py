import gradio as gr

def info():
    return "This Space shows the Reinforcement Learning Mario Agent. The actual game runs locally, but this demo displays a pre-recorded gameplay clip."

with gr.Blocks() as demo:
    gr.Markdown("# 🎮 Super Mario RL Agent Demo")
    gr.Markdown(
        """
        This is a Reinforcement Learning agent trained to play Super Mario Bros using Deep RL.
        Below is a recorded gameplay clip from the trained model.
        """
    )

    gr.Markdown("### 🕹️ Agent Gameplay")
    gr.Video("videos/demo.mp4")   # or .gif if you have that

    gr.Markdown("### ℹ️ Project Info")
    btn = gr.Button("About this project")
    out = gr.Textbox()

    btn.click(info, None, out)

demo.launch()
