import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";
import { vectorStore } from "./vectorStore.js";
import  data  from "./data.js";

export const addDocumentsToVectorStore = async (video) => {

    const docs = [
        new Document({
            pageContent: video.transcript,
            metadata: {
                video_id: video.video_id
            }
        })
    ];

    const splitter =
        new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200
        });

    const chunks =
        await splitter.splitDocuments(docs);

    await vectorStore.addDocuments(chunks);

    console.log(`Stored ${chunks.length} chunks successfully.`);
};

for (const video of data) {
    await addDocumentsToVectorStore(video);
}
console.log("✅ Ingestion completed.");