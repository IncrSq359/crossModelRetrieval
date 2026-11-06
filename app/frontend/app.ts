const textInput = document.getElementById("text-query") as HTMLInputElement;
const searchBtn = document.getElementById("btn-search-text");
if (searchBtn === null) {
    throw new Error("找不到搜索按钮，检查id");
}
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
    renderResults(data);
    renderStatus(data);
})

const resultsBox = document.getElementById("results");

function renderResults(data: SearchResponse): void{
    if (resultsBox === null) {
        throw new Error("找不到结果容器，检查id");
    }
    resultsBox.innerHTML = "";
    for (const item of data.results) {
        const card = document.createElement("div");
        card.className = "card";

        const img = document.createElement("img");
        img.src = item.url;
        img.alt = item.filename;

        const score = document.createElement("p");
        score.textContent = "相似度：" + item.score.toFixed(4);
        
        const captionList = document.createElement("ul");
        for (const caption of item.captions) {
            const li = document.createElement("li");
            li.textContent = caption;
            captionList.appendChild(li);
        }

        card.appendChild(img);
        card.appendChild(score);
        card.appendChild(captionList);
        resultsBox.appendChild(card);
    }
}

const statusBar = document.getElementById("status-bar");
function renderStatus(data: SearchResponse): void{
    if (statusBar === null) {
        throw new Error("找不到状态栏，检查id");
    }
    statusBar.textContent =
    "耗时：" + data.latency_ms + " ms　检索后端：" + data.retriever + "　结果数：" + data.results.length;
}

async function searchByImage(file:File):Promise<SearchResponse> {
    const formData = new FormData();
    formData.append("file", file);
    const resp = await fetch("/api/search/image", {
        method: "POST",
        body: formData,
    });
    if (!resp.ok) {
        throw new Error(`请求失败:${resp.status}`);
    }
    return (await resp.json()) as SearchResponse;
}

const imageInput = document.getElementById("image-file") as HTMLInputElement;
imageInput.addEventListener("change", async () => {
    const file = imageInput.files?.[0];
    if (file === undefined) {
        return;
    }
    const data =await searchByImage(file);
    renderResults(data);
    renderStatus(data);
})