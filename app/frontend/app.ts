const textInput = document.getElementById("text-query") as HTMLInputElement;
const searchBtn = document.getElementById("btn-search-text");
if (searchBtn === null) {
    throw new Error("找不到搜索按钮，检查id");
}
searchBtn.addEventListener("click", () => {
    const query: string = textInput.value;
    console.log("用户输入：", query);
});
