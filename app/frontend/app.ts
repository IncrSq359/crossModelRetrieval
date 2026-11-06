const textInput = document.getElementById("text-query") as HTMLInputElement;
const searchBtn = document.getElementById("btn-search-text");
if (searchBtn === null) {
    throw new Error("找不到搜索按钮，检查id");
}
searchBtn.addEventListener("click", () => {
    const query: string = textInput.value;
    console.log("用户输入：", query);
});

// 一条检索结果（字段和后端 API 契约一致）
interface SearchResult{
    filename: string;
    score: number;
    url: string;
    captions: string[];
}

// 整个响应
interface SearchResponse{
    latency_ms: number;
    retriever: string;
    results: SearchResult[];
}

//发送请求返回结果
async function searchByText(query: string): Promise<SearchResponse> {
    const response = await fetch("/api/search/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query, top_k: 20 }),
    });
    return await response.json()
}

searchBtn.addEventListener("click", async () => {
    const query: string = textInput.value;
    const data = await searchByText(query);
    console.log("检索结果：", data);
})

