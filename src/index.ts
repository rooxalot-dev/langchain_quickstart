import { config } from 'dotenv';

import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { StringOutputParser } from "@langchain/core/output_parsers";

import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";

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

  // Prompt
  const historyAwareRetrievalPrompt = ChatPromptTemplate.fromMessages([
    [
      'system',
      `Answer the user's questions based on the below context:

      {context}
      `,
    ],
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
  ]);

  // Retriever (both from vectorstore and chat history)
  const historyAwareRetrieverChain = await createHistoryAwareRetriever({
    llm: chatModel,
    retriever: vectorStoreRetriever,
    rephrasePrompt: historyAwareRetrievalPrompt,
  });

  // Combined docs chain
  const historyAwareCombineDocumentChain = await createStuffDocumentsChain({
    llm: chatModel,
    prompt: historyAwareRetrievalPrompt,
  });

  // Retrieval chain
  const conversationalRetrievalChain = await createRetrievalChain({
    retriever: vectorStoreRetriever,
    combineDocsChain: historyAwareCombineDocumentChain,
  });

  const conversationalRetrievalChainAnswer = await conversationalRetrievalChain.invoke({
    chat_history: [
      new HumanMessage("Can LangSmith help test my LLM applications?"),
      new AIMessage("Yes!"),
    ],
    input: 'Tell me how!',
  });

  console.log('invokeChat -> Response', { conversationalRetrievalChainAnswer });
}

invokeChat();
