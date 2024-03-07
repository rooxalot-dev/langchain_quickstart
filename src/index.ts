import { config } from 'dotenv';

import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatOpenAI } from '@langchain/openai';

config();

const invokeChat = async () => {
    // Obtains OPENAI_API_KEY from environment variables
    const chatModel = new ChatOpenAI();

    const prompt = ChatPromptTemplate.fromMessages([
        ["system", "You are a world class technical documentation writer."],
        ["user", "{input}"],
      ]);

    const chain = prompt.pipe(chatModel);
    const chatResponse = await chain.invoke({ input: 'What is LangSmith?' });
    console.log('invokeChat', { chatResponse });
}

invokeChat();