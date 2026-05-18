from typing import Optional, Tuple

from loguru import logger

from tau2.data_model.message import (
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    UserMessage,
)
from tau2.data_model.tasks import UserInstructions
from tau2.environment.tool import Tool
from tau2.user.base import (
    OUT_OF_SCOPE,
    STOP,
    TRANSFER,
    BaseUser,
    UserState,
    ValidUserInputMessage,
    is_valid_user_history_message,
)
from tau2.utils import DATA_DIR
from tau2.utils.llm_utils import generate

GLOBAL_USER_SIM_GUIDELINES_DIR = DATA_DIR / "tau2" / "user_simulator"


GLOBAL_USER_SIM_GUIDELINES_PATH = (
    GLOBAL_USER_SIM_GUIDELINES_DIR / "simulation_guidelines.md"
)

GLOBAL_USER_SIM_GUIDELINES_PATH_TOOLS = (
    GLOBAL_USER_SIM_GUIDELINES_DIR / "simulation_guidelines_tools.md"
)


def get_global_user_sim_guidelines(use_tools: bool = False) -> str:
    """
    Get the global user simulator guidelines.

    Args:
        use_tools: Whether to use the tools guidelines.

    Returns:
        The global user simulator guidelines.
    """
    if use_tools:
        with open(GLOBAL_USER_SIM_GUIDELINES_PATH_TOOLS, "r") as fp:
            user_sim_guidelines = fp.read()
    else:
        with open(GLOBAL_USER_SIM_GUIDELINES_PATH, "r") as fp:
            user_sim_guidelines = fp.read()
    return user_sim_guidelines


SYSTEM_PROMPT = """
{global_user_sim_guidelines}

<scenario>
{instructions}
</scenario>
""".strip()

SYSTEM_PROMPT_SUMMARY = """
{global_user_sim_guidelines}

<scenario>
{instructions}
</scenario>

Please summary the current state of the conversation and make decision based on the state.
""".strip()


class UserSimulator(BaseUser):
    """Stateless implementation of a user simulator."""

    def __init__(
        self,
        tools: Optional[list[Tool]] = None,
        instructions: Optional[UserInstructions] = None,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
    ):
        super().__init__(instructions=instructions, llm=llm, llm_args=llm_args)
        self.tools = tools

    @property
    def global_simulation_guidelines(self) -> str:
        """
        The simulation guidelines for the user simulator.
        """
        use_tools = self.tools is not None
        return get_global_user_sim_guidelines(use_tools=use_tools)

    @property
    def system_prompt(self) -> str:
        """
        The system prompt for the user simulator.
        """
        if self.instructions is None:
            logger.warning("No instructions provided for user simulator")

        if self.llm_args and self.llm_args.get("summary", False):
            system_prompt = SYSTEM_PROMPT_SUMMARY.format(
                global_user_sim_guidelines=self.global_simulation_guidelines,
                instructions=self.instructions,
            )
        else:
            system_prompt = SYSTEM_PROMPT.format(
                global_user_sim_guidelines=self.global_simulation_guidelines,
                instructions=self.instructions,
            )
        return system_prompt

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> UserState:
        """
        Get the initial state of the user simulator.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_user_history_message(m) for m in message_history), (
            "Invalid user message history. User messages must be of type UserMessage, AssistantMessage, or ToolMessage to User."
        )

        user_state = UserState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )
        return user_state

    @classmethod
    def is_stop(cls, message: UserMessage) -> bool:
        """
        Check if the message is a stop message.
        """
        if message.is_tool_call():
            return False
        assert message.content is not None
        return (
            STOP in message.content
            or TRANSFER in message.content
            or OUT_OF_SCOPE in message.content
        )

    def generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        return self._generate_next_message(message, state)

    def _generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        """Get the response from the user simulator.

        Args:
            message: The assistant or tool message.
            state: The user simulator's state.

        Returns:
            A tuple containing the user message and the updated user state.
        """
        # Updating state with new message
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        messages = state.system_messages + state.flip_roles()

        # Generate response
        # Filter out summary parameter as it's only used for system prompt generation
        llm_kwargs = {k: v for k, v in self.llm_args.items() if k != "summary"}
        
        assistant_message = generate(
            model=self.llm,
            messages=messages,
            tools=self.tools,
            use_azure=True,
            use_hf=False,
            **llm_kwargs,
        )

        user_response = assistant_message.content
        logger.debug(f"Response: {user_response}")

        user_message = UserMessage(
            role="user",
            content=user_response,
            cost=assistant_message.cost,
            usage=assistant_message.usage,
            raw_data=assistant_message.raw_data,
        )

        # flip the requestor of the tool calls
        if assistant_message.tool_calls is not None:
            user_message.tool_calls = []
            for tool_call in assistant_message.tool_calls:
                user_message.tool_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                        requestor="user",
                    )
                )

        # Updating state with response
        state.messages.append(user_message)
        return user_message, state


class DummyUser(UserSimulator):
    """A dummy user to run a agent solo simulation."""

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> UserState:
        return UserState(messages=[], system_messages=[])

    def is_stop(cls, message: UserMessage) -> bool:
        raise NotImplementedError("DummyUser does not support stop messages")

    def set_seed(self, seed: int):
        pass

    def generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> tuple[UserMessage, UserState]:
        # 在 solo 模式下，DummyUser 不应该被调用
        # 但在非 solo 模式下，我们需要提供一个基本的实现
        # 创建一个简单的用户消息作为响应
        user_message = UserMessage(
            role="user",
            content="I understand. Please continue.",
            cost=0.0,
            usage=None,
            raw_data=None,
        )
        
        # 更新状态
        if state.messages is None:
            state.messages = []
        state.messages.append(user_message)
        
        return user_message, state
