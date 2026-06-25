import OpenAI from "openai";
import dotenv from "dotenv";
import data from './data.js';
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";
import { meta } from "zod/v4/core";
import { OpenAIEmbeddings, ChatOpenAI } from '@langchain/openai';
import { MemoryVectorStore } from '@langchain/classic/vectorstores/memory';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';
dotenv.config();

const video1 = data[0];
const docs = [new Document({ pageContent: video1.transcript, metadata: { video_id: video1.video_id } })];

//Split video into chunk
const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200
})

const chunks = await textSplitter.splitDocuments(docs);
//console.log(chunks);

const client = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

const embeddings = new OpenAIEmbeddings({
    model: 'text-embedding-3-small'
});

const vectorStore = new MemoryVectorStore(embeddings);

await vectorStore.addDocuments(chunks);
//retrieve the most relevant docs
const retrieveTool = tool(async ({ query }) => {
    const relevantDocs =
        await vectorStore.similaritySearch(
            query,
            3
        );

    return relevantDocs
        .map(doc => doc.pageContent)
        .join("\n\n");
}, {
    name: 'retrieve',
    description: 'Search the video transcript for relevant information.',
    schema: z.object({
        query: z.string(),
    })
})

const model = new ChatOpenAI({
    model: "gpt-4o-mini"
});

const modelWithTools = model.bindTools([
    retrieveTool
]);

const response = await modelWithTools.invoke(
    "Which word speaker using frequently?"
);
console.log(JSON.stringify(response, null, 2));
const toolCall = response.tool_calls[0];

const toolResult = await retrieveTool.invoke(
    toolCall.args
);

const finalResponse =
    await modelWithTools.invoke([
        {
            role: "user",
            content: "Which word speaker using frequently?"
        },
        response,
        {
            role: "tool",
            content: toolResult,
            tool_call_id: toolCall.id
        }
    ]);

console.log(finalResponse.content);

