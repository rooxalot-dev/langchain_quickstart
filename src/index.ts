import { config } from 'dotenv';

import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from "@langchain/core/output_parsers";

import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";

config();

const invokeChat = async () => {
  // Obtains OPENAI_API_KEY from environment variables
  const chatModel = new ChatOpenAI();

  const loader = new CheerioWebBaseLoader("https://docs.smith.langchain.com/user_guide");
  const docs = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter();
  const splitDocs = await splitter.splitDocuments(docs);

  console.log('Loaded context splitted doc', {
    splits: splitDocs.length,
    contentLength: splitDocs[0].pageContent.length,
  });

  const embeddings = new OpenAIEmbeddings();
  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );
  const vectorStoreRetriever = vectorStore.asRetriever();

  const prompt = ChatPromptTemplate.fromTemplate(`
    Answer the following question basend only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}
  `);

  const documentChain = await createStuffDocumentsChain({
    llm: chatModel,
    prompt,
    outputParser: new StringOutputParser(),
  });

  const retrievalChain = await createRetrievalChain({
    combineDocsChain: documentChain,
    retriever: vectorStoreRetriever,
  });

  const answerResponse = await retrievalChain.invoke({ input: 'what is LangSmith?' });

  console.log('invokeChat -> Response', { answerResponse });
}

invokeChat();
