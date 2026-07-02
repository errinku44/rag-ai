import dotenv from "dotenv";
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { vectorStore } from "./vectorStore.js";

dotenv.config();

const retrieveTool = tool(
    async ({ query }) => {
        const relevantDocs = await vectorStore.similaritySearch(
            query,
            3
        );

        return relevantDocs
            .map((doc) => doc.pageContent)
            .join("\n\n");
    },
    {
        name: "retrieve",
        description:
            "Search the video transcript for relevant information.",
        schema: z.object({
            query: z.string(),
        }),
    }
);

const model = new ChatOpenAI({
    model: "gpt-4o-mini",
});

const modelWithTools = model.bindTools([retrieveTool]);

const question = process.argv.slice(2).join(" ");
if (!question) {
    console.log("Usage:");
    console.log('node agent.js "What is this video about?"');
    process.exit(1);
}

const response = await modelWithTools.invoke([
    {
        role: "system",
        content:
            "You must use the retrieve tool before answering. Never answer from your own knowledge.",
    },
    {
        role: "user",
        content: question,
    },
]);

console.log("=========== FIRST RESPONSE ===========");
console.log(JSON.stringify(response, null, 2));

if (!response.tool_calls?.length) {
    console.log("\nNo tool call was made.\n");
    console.log(response.content);
    process.exit(0);
}

const toolCall = response.tool_calls[0];

const toolResult = await retrieveTool.invoke(toolCall.args);

const finalResponse = await modelWithTools.invoke([
    {
        role: "system",
        content:
            "Answer only using the information returned by the retrieve tool.",
    },
    {
        role: "user",
        content: question,
    },
    response,
    {
        role: "tool",
        content: toolResult,
        tool_call_id: toolCall.id,
    },                      
]);

console.log("\n=========== FINAL ANSWER ===========");
console.log(finalResponse.content);