import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import pathlib
from abc import ABC, abstractmethod

load_dotenv()


class Chat(ABC):
    def __init__(self, session_key_prefix="default", model=None, pdf_path=None):
        """
        Initialize base Chat component.

        Args:
            session_key_prefix (str): Prefix for session state keys
            model (str): Model identifier
            pdf_path (str, optional): Path to a PDF file for context
        """
        self.session_key_prefix = session_key_prefix
        self.messages_key = f"{session_key_prefix}_messages"
        self.model = model
        self.pdf_path = pdf_path

        # Initialize message history with system message
        if self.messages_key not in st.session_state:
            system_message = (
                "You are an expert ML researcher having a technical discussion with "
                "another ML expert about a research paper. Your role is to analyze "
                "the paper in detail and engage in a sophisticated technical dialogue. "
                "Use technical ML terminology freely, as the user is well-versed in ML. "
                "Focus on providing deep insights, discussing methodological choices, "
                "and critically analyzing the paper's contributions and limitations. "
                "When appropriate, relate the paper's content to other relevant research "
                "in the field."
            )
            st.session_state[self.messages_key] = [
                {"role": "system", "content": system_message}
            ]

    def render(self, chat_title="Chat"):
        """
        Render the chat interface.

        Args:
            chat_title (str): Title to display above the chat
        """
        # Create a row for the title and clear button
        title_col, clear_col = st.columns([6, 1])
        with title_col:
            st.subheader(chat_title)
        with clear_col:
            if st.button(
                "🗑️", key=f"{self.session_key_prefix}_clear", help="Clear chat history"
            ):
                self.clear_history()
                st.rerun()

        # Display existing messages
        for message in st.session_state[self.messages_key]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("What would you like to know?"):
            # Add user message
            self._add_user_message(prompt)
            # Generate and display assistant response
            self._generate_response(prompt)

    def _add_user_message(self, prompt):
        """Add a user message to the chat history."""
        st.session_state[self.messages_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    @abstractmethod
    def _generate_response(self, prompt):
        """Generate and display assistant response."""
        pass

    @abstractmethod
    def clear_history(self):
        """Clear the chat history."""
        pass


class OpenAIChat(Chat):
    def __init__(self, session_key_prefix="default", model="gpt-3.5-turbo"):
        """Initialize OpenAI Chat component."""
        super().__init__(session_key_prefix, model)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.openai_api_key)

    def _generate_response(self, prompt):
        """Generate and display assistant response using OpenAI."""
        with st.chat_message("assistant"):
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state[self.messages_key]
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state[self.messages_key].append(
            {"role": "assistant", "content": response}
        )

    def clear_history(self):
        """Clear the chat history."""
        st.session_state[self.messages_key] = []


class GeminiChat(Chat):
    def __init__(
        self, session_key_prefix="default", model="gemini-2.0-flash", pdf_path=None
    ):
        """Initialize Gemini Chat component."""
        super().__init__(session_key_prefix, model, pdf_path)
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.gemini_api_key)
        self.chat_key = f"{session_key_prefix}_chat"

        # Initialize chat session
        if self.chat_key not in st.session_state:
            st.session_state[self.chat_key] = self.client.chats.create(model=self.model)

    def _generate_response(self, prompt):
        """Generate and display assistant response using Gemini."""
        with st.chat_message("assistant"):
            if self.pdf_path:
                # If PDF context is provided, use content generation API
                pdf_file = pathlib.Path(self.pdf_path)
                system_context = st.session_state[self.messages_key][0]["content"]
                response_stream = self.client.models.generate_content_stream(
                    model=self.model,
                    contents=[
                        types.Part.from_bytes(
                            data=pdf_file.read_bytes(),
                            mime_type="application/pdf",
                        ),
                        f"{system_context}\n\nBased on the PDF content above, please respond to: {prompt}",
                    ],
                )
                response = st.write_stream((chunk.text for chunk in response_stream))
            else:
                # Use regular chat API if no PDF context
                response_stream = st.session_state[self.chat_key].send_message_stream(
                    prompt
                )
                response = st.write_stream((chunk.text for chunk in response_stream))
        st.session_state[self.messages_key].append(
            {"role": "assistant", "content": response}
        )

    def clear_history(self):
        """Clear the chat history and create a new chat session."""
        st.session_state[self.messages_key] = []
        st.session_state[self.chat_key] = self.client.chats.create(model=self.model)
