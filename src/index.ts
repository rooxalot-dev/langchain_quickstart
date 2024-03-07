import { config } from 'dotenv';

import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatOpenAI } from '@langchain/openai';
import { StringOutputParser } from "@langchain/core/output_parsers";

config();

const invokeChat = async () => {
    // Obtains OPENAI_API_KEY from environment variables
    const chatModel = new ChatOpenAI();

    const prompt = ChatPromptTemplate.fromMessages([
        ["system", "You are a world class technical documentation writer with relevant knowledge on technology."],
        ["user", "{input}"],
      ]);
    const outputParser = new StringOutputParser();

    const chain = prompt
        .pipe(chatModel)
        .pipe(outputParser);

    const chatResponse = await chain.invoke({ input: 'What is LangSmith?' });
    console.log('invokeChat', { chatResponse });
}

invokeChat();