import { OpenAIEmbeddings } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";

const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
});

export const vectorStore =
    await QdrantVectorStore.fromExistingCollection(
        embeddings,
        {
            url: "http://localhost:6333",
            collectionName: "youtube-transcripts",
        }
    );