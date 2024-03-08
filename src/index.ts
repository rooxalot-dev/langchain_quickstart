import { config } from 'dotenv';

import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from "@langchain/core/output_parsers";

import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

config();

const invokeChat = async () => {
  // Obtains OPENAI_API_KEY from environment variables
  const chatModel = new ChatOpenAI();

  const loader = new CheerioWebBaseLoader("https://docs.smith.langchain.com/user_guide");
  const docs = await loader.load();
  const [{ pageContent }] = await loader.load();
  console.log('Loaded context doc', {
    length: pageContent.length,
    contentLength: pageContent.length,
  });

  const splitter = new RecursiveCharacterTextSplitter();
  const splitDocs = await splitter.splitDocuments(docs);
  console.log('Loaded context splitted doc', {
    splits: splitDocs.length,
    contentLength: splitDocs[0].pageContent.length,
  });

  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a world class technical documentation writer with relevant knowledge on technology."],
    ["user", "{input}"],
  ]);
  const outputParser = new StringOutputParser();

  const chain = prompt
    .pipe(chatModel)
    .pipe(outputParser);

  //const chatResponse = await chain.invoke({ input: 'What is LangSmith?' });
  //console.log('invokeChat', { chatResponse });
}

invokeChat();
