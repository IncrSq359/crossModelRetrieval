"use strict";
const textInput = document.getElementById("text-query");
const searchBtn = document.getElementById("btn-search-text");
if (searchBtn === null) {
    throw new Error("找不到搜索按钮，检查id");
}
searchBtn.addEventListener("click", () => {
    const query = textInput.value;
    console.log("用户输入：", query);
});
//发送请求返回结果
async function searchByText(query) {
    const response = await fetch("/api/search/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query, top_k: 20 }),
    });
    return await response.json();
}
searchBtn.addEventListener("click", async () => {
    const query = textInput.value;
    const data = await searchByText(query);
    console.log("检索结果：", data);
});
