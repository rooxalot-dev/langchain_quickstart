import { config } from 'dotenv';

import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';
import { HumanMessage, AIMessage } from "@langchain/core/messages";

import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";

import { createRetrieverTool } from "langchain/tools/retriever";

import { TavilySearchResults } from "@langchain/community/tools/tavily_search";

config();

const invokeChat = async () => {
  // Obtains OPENAI_API_KEY from environment variable
  const chatModel = new ChatOpenAI({
    modelName: "gpt-3.5-turbo-1106",
    temperature: 0.3,
  });

  // Setup the vectorstore retriever
  const loader = new CheerioWebBaseLoader("https://docs.smith.langchain.com/user_guide");
  const docs = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter();
  const splitDocs = await splitter.splitDocuments(docs);

  const embeddings = new OpenAIEmbeddings();
  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );
  const vectorStoreRetriever = vectorStore.asRetriever();

  // instantiates Tavily search engine and obtains TAVILY_API_KEY from environment variable
  const searchTool = new TavilySearchResults();

  const retrieverTool = createRetrieverTool(vectorStoreRetriever, {
    name: "langsmith_search",
    description: "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
  });

  // List available tools
  const tools = [retrieverTool, searchTool];

}

invokeChat();
