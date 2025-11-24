from typing import Any, AsyncIterator

from chatkit.server import ChatKitServer
from chatkit.store import AttachmentStore, Store, StoreItemType
from chatkit.types import (
    ThreadMetadata,
    ThreadStreamEvent,
    UserMessageItem,
)
from agents import Runner
from chatkit.agents import AgentContext
from agents import Agent
from chatkit.agents import stream_agent_response
from openai.types.responses.easy_input_message_param import EasyInputMessageParam

async def to_input_item(input):
    return [EasyInputMessageParam(
        content= input.content[0].text,
        role="user",
        type="message"
    )]


class MyChatKitServer(ChatKitServer):
    def __init__(
        self, data_store: Store, attachment_store: AttachmentStore | None = None
    ):
        super().__init__(data_store, attachment_store)

    assistant_agent = Agent[AgentContext](
        model="gpt-4.1",
        name="Assistant",
        instructions="You are a helpful assistant",
    )

    async def respond(
        self,
        thread: ThreadMetadata,
        input: UserMessageItem | None,
        context: Any,
    ) -> AsyncIterator[ThreadStreamEvent]:
        print("Generating response...")
        agent_context = AgentContext(
            thread=thread,
            store=self.store,
            request_context=context,
        )
        result = Runner.run_streamed(
            self.assistant_agent,
            await to_input_item(input),
            context=agent_context,
        )
        async for event in stream_agent_response(agent_context, result):
            yield event
