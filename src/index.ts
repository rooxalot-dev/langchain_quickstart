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

  const prompt = ChatPromptTemplate.fromTemplate(`
    Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    You don't need to explain that the answer that you'll give is based on the context.

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

  // Conversational Chain

  const historyAwarePrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
    [
      "user",
      "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
    ],
  ]);

  const historyAwareRetrieverChain = await createHistoryAwareRetriever({
    llm: chatModel,
    retriever: vectorStoreRetriever,
    rephrasePrompt: historyAwarePrompt,
  });

  const mockChatHistory = [
    new HumanMessage("Can LangSmith help test my LLM applications?"),
    new AIMessage("Yes!"),
  ];

  const historyAwareAnswer = await historyAwareRetrieverChain.invoke({
    chat_history: mockChatHistory,
    input: 'Tell me how!'
  });

  console.log('invokeChat -> Response', { historyAwareAnswer });


}

invokeChat();
