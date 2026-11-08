"use strict";
const textInput = document.getElementById("text-query");
const searchBtn = document.getElementById("btn-search-text");
if (searchBtn === null) {
    throw new Error("找不到搜索按钮，检查id");
}
//发送请求返回结果
async function searchByText(query) {
    const response = await fetch("/api/search/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query, top_k: 20 }),
    });
    if (!response.ok) {
        throw new Error(`请求失败:${response.status}`);
    }
    return await response.json();
}
searchBtn.addEventListener("click", async () => {
    const query = textInput.value;
    try {
        const data = await searchByText(query);
        renderResults(data);
        renderStatus(data);
    }
    catch (error) {
        handleError(error);
    }
});
const resultsBox = document.getElementById("results");
function renderResults(data) {
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
        img.addEventListener("click", () => {
            openLightbox(item.url);
        });
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
function renderStatus(data) {
    if (statusBar === null) {
        throw new Error("找不到状态栏，检查id");
    }
    statusBar.textContent =
        "耗时：" + data.latency_ms + " ms　检索后端：" + data.retriever + "　结果数：" + data.results.length;
}
const lightbox = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightbox-img");
function openLightbox(url) {
    if (lightbox === null) {
        throw new Error("找不到灯箱，检查id");
    }
    lightboxImg.src = url;
    lightbox.classList.add("show");
}
function closeLightbox() {
    if (lightbox === null) {
        throw new Error("找不到灯箱，检查id");
    }
    lightbox.classList.remove("show");
}
lightbox?.addEventListener("click", closeLightbox);
function renderError(message) {
    if (statusBar === null) {
        throw new Error("找不到状态栏，检查id");
    }
    statusBar.textContent = "出错了:" + message;
}
function handleError(error) {
    if (error instanceof Error) {
        renderError(error.message);
    }
    else {
        String(error);
    }
}
async function searchByImage(file) {
    const formData = new FormData();
    formData.append("file", file);
    const resp = await fetch("/api/search/image", {
        method: "POST",
        body: formData,
    });
    if (!resp.ok) {
        throw new Error(`请求失败:${resp.status}`);
    }
    return (await resp.json());
}
const imageInput = document.getElementById("image-file");
imageInput.addEventListener("change", async () => {
    const file = imageInput.files?.[0];
    if (file === undefined) {
        return;
    }
    try {
        const data = await searchByImage(file);
        renderResults(data);
        renderStatus(data);
    }
    catch (error) {
        handleError(error);
    }
});
