import gradio as gr
from llama_parse import LlamaParse
from app.inference.inference import Inference
from configuration import config_parser
import time

# inspired by https://www.gradio.app/guides/creating-a-custom-chatbot-with-blocks

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def add_message(history, message):
    if (message["files"] is not None and message["text"] != ''):
        for x in message["files"] :
            history.append({"role": "user", "content":{"file": x, "content":message["text"] }})
            return history, gr.MultimodalTextbox(value=None, interactive=False)
    
    for x in message["files"]:
        history.append({"role": "user", "content": {"path": x}})
    if message["text"] != '':
        history.append({"role": "user", "content": message["text"]})
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def bot(history: list):
    print(history)
    parser = config_parser()
    args = parser.parse_args()

    agent = Inference(args=args)

    query = history[-1]["content"]  # TODO : sanitize input ?
    print(query,type(query))
    if query[0].endswith(".pdf"):
        parser = LlamaParse(result_type="markdown", api_key=args.llama_parse_key)
        documents = parser.load_data(query[0])
        parsed_doc = ""
        for doc in documents :
            parsed_doc += doc.text
            parsed_doc += "\n"
        print(parsed_doc)
        query = parsed_doc

    response = agent.query_llm(question=query)

    answer = "The problem you are facing is probably : " + "\n"

    for source_node in response.source_nodes:
        score = source_node.score
        name = source_node.metadata["model_name"]
        # name = source_node.metadata["problem_family"]
        source_code = source_node.metadata["source_code"]
        answer += str(name) + " with a score of " + str(score) + "\n"

    history.append({"role": "assistant", "content": ""})
    for character in answer:
        history[-1]["content"] += character
        time.sleep(0.03)
        yield history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, type="messages")

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_count="multiple",
        placeholder="Enter message or upload file...",
        show_label=False,
        file_types = [".pdf"],
    )

    chat_msg = chat_input.submit(
        add_message, [chatbot, chat_input], [chatbot, chat_input]
    )
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    chatbot.like(print_like_dislike, None, None, like_user_message=True)

demo.launch(share=False)
