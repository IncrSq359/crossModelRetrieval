const textInput = document.getElementById("text-query") as HTMLInputElement;
const searchBtn = document.getElementById("btn-search-text") as HTMLButtonElement | null;
if (searchBtn === null) {
    throw new Error("找不到搜索按钮，检查id");
}
if (textInput === null) {
    throw new Error("找不到输入框，检查id");
}

// 一条检索结果（字段和后端 API 契约一致，字段名不可改动）
interface SearchResult {
    filename: string;
    score: number;
    url: string;
    captions: string[];
}

// 整个响应
interface SearchResponse {
    latency_ms: number;
    retriever: string;
    results: SearchResult[];
}

// 以下容器在 index.html 中固定存在，id 不可改动
const resultsBox = document.getElementById("results");
const statusBar = document.getElementById("status-bar");
const lightbox = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightbox-img") as HTMLImageElement | null;
const lightboxCaption = document.getElementById("lightbox-caption");
const imageInput = document.getElementById("image-file") as HTMLInputElement | null;
const dropzone = document.getElementById("image-dropzone");
const dropzoneHint = document.getElementById("dropzone-hint");

// 增强交互元素（可选存在）
const themeToggle = document.getElementById("theme-toggle") as HTMLButtonElement | null;
const queryPreview = document.getElementById("query-preview") as HTMLImageElement | null;
const lightboxStage = document.getElementById("lightbox-stage");
const lightboxCloseBtn = document.getElementById("lightbox-close") as HTMLButtonElement | null;
const lightboxPrevBtn = document.getElementById("lightbox-prev") as HTMLButtonElement | null;
const lightboxNextBtn = document.getElementById("lightbox-next") as HTMLButtonElement | null;

let requestSeq = 0;
let currentResults: SearchResult[] = [];
let currentLightboxIndex = -1;
let previewObjectUrl: string | null = null;

/* ---------- 主题切换 ---------- */

function initTheme(): void {
    try {
        const saved = localStorage.getItem("cmr-theme");
        if (saved === "light" || saved === "dark") {
            document.documentElement.setAttribute("data-theme", saved);
        }
    } catch {
        // 忽略存储受限环境
    }
}

function toggleTheme(): void {
    const root = document.documentElement;
    const current = root.getAttribute("data-theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const isDark = current === "dark" || (current === null && prefersDark);
    const next = isDark ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try {
        localStorage.setItem("cmr-theme", next);
    } catch {
        // 忽略存储受限环境
    }
}

themeToggle?.addEventListener("click", toggleTheme);
initTheme();

/* ---------- 请求 ---------- */

async function searchByText(query: string): Promise<SearchResponse> {
    const response = await fetch("/api/search/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query, top_k: 20 }),
    });
    if (!response.ok) {
        throw new Error(`请求失败 ${response.status}`);
    }
    return (await response.json()) as SearchResponse;
}

async function searchByImage(file: File): Promise<SearchResponse> {
    const formData = new FormData();
    formData.append("file", file);
    const resp = await fetch("/api/search/image", {
        method: "POST",
        body: formData,
    });
    if (!resp.ok) {
        throw new Error(`请求失败 ${resp.status}`);
    }
    return (await resp.json()) as SearchResponse;
}

/* ---------- 状态 ---------- */

function setStatus(text: string, isError?: boolean): void {
    if (statusBar === null) {
        return;
    }
    statusBar.textContent = text;
    statusBar.classList.toggle("is-error", isError === true);
}

function describeError(error: unknown): string {
    if (error instanceof Error) {
        return error.message;
    }
    return String(error);
}

/* ---------- 结果 ---------- */

function clearResults(): void {
    if (resultsBox !== null) {
        resultsBox.innerHTML = "";
    }
}

function renderSkeletons(count: number): void {
    clearResults();
    if (resultsBox === null) {
        return;
    }
    const frag = document.createDocumentFragment();
    for (let i = 0; i < count; i++) {
        const card = document.createElement("div");
        card.className = "skeleton";
        card.setAttribute("aria-hidden", "true");

        const media = document.createElement("div");
        media.className = "skeleton-media";

        const body = document.createElement("div");
        body.className = "skeleton-body";

        const line1 = document.createElement("div");
        line1.className = "skeleton-line";

        const line2 = document.createElement("div");
        line2.className = "skeleton-line short";

        body.appendChild(line1);
        body.appendChild(line2);
        card.appendChild(media);
        card.appendChild(body);
        frag.appendChild(card);
    }
    resultsBox.appendChild(frag);
}

function renderEmpty(message: string): void {
    clearResults();
    currentResults = [];
    if (resultsBox === null) {
        return;
    }
    const box = document.createElement("div");
    box.className = "empty";

    const icon = document.createElement("span");
    icon.className = "empty-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.innerHTML =
        '<svg viewBox="0 0 24 24" width="24" height="24" fill="none">' +
        '<rect x="3" y="5" width="18" height="14" rx="2.5" stroke="currentColor" stroke-width="1.6"/>' +
        '<circle cx="8.5" cy="10" r="1.5" fill="currentColor"/>' +
        '<path d="M4.5 17l5-4.8 3.2 3.1 2.4-2.3 4.4 4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>' +
        "</svg>";

    const p = document.createElement("p");
    p.textContent = message;

    box.appendChild(icon);
    box.appendChild(p);
    resultsBox.appendChild(box);
}

function buildShot(item: SearchResult, index: number): HTMLElement {
    const rawScore = Number(item.score);
    const score = isFinite(rawScore) ? rawScore : 0;
    const captions = Array.isArray(item.captions) ? item.captions : [];

    const figure = document.createElement("figure");
    figure.className = "shot";
    figure.style.animationDelay = `${Math.min(index, 11) * 24}ms`;

    const media = document.createElement("div");
    media.className = "shot-media";

    const img = document.createElement("img");
    img.src = item.url;
    img.alt = captions.length > 0 ? captions[0] : item.filename;
    img.loading = "lazy";
    img.decoding = "async";
    img.tabIndex = 0;
    img.setAttribute("role", "button");
    img.setAttribute("aria-label", "查看大图：" + item.filename);
    img.addEventListener("click", () => {
        openLightboxAt(index);
    });
    img.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            openLightboxAt(index);
        }
    });

    media.appendChild(img);

    const caption = document.createElement("figcaption");

    const meta = document.createElement("div");
    meta.className = "shot-meta";

    const rank = document.createElement("span");
    rank.className = "shot-rank";
    rank.textContent = String(index + 1).padStart(2, "0");

    const scoreEl = document.createElement("span");
    scoreEl.className = "shot-score";
    scoreEl.textContent = score.toFixed(3);

    meta.appendChild(rank);
    meta.appendChild(scoreEl);

    const text = document.createElement("p");
    text.className = "cap";
    text.textContent = captions.length > 0 ? captions[0] : item.filename;

    const file = document.createElement("p");
    file.className = "file";
    file.textContent = item.filename;
    file.title = item.filename;

    caption.appendChild(meta);
    caption.appendChild(text);
    caption.appendChild(file);
    figure.appendChild(media);
    figure.appendChild(caption);
    return figure;
}

function renderResults(data: SearchResponse): void {
    const results = Array.isArray(data.results) ? data.results : [];
    if (results.length === 0) {
        renderEmpty("没有找到匹配的图片。换个说法，或者换一张图试试。");
        return;
    }

    currentResults = results;
    clearResults();
    if (resultsBox === null) {
        throw new Error("找不到结果容器，检查id");
    }
    const frag = document.createDocumentFragment();
    results.forEach((item, index) => {
        frag.appendChild(buildShot(item, index));
    });
    resultsBox.appendChild(frag);
}

function renderStatus(data: SearchResponse): void {
    const count = Array.isArray(data.results) ? data.results.length : 0;
    if (count === 0) {
        setStatus("没有找到匹配的图片，用时 " + data.latency_ms + " 毫秒");
        return;
    }
    setStatus("共 " + count + " 张 · 用时 " + data.latency_ms + " 毫秒 · " + String(data.retriever).toUpperCase());
}

/* ---------- 检索流程 ---------- */

function setBusy(busy: boolean): void {
    if (searchBtn === null) {
        return;
    }
    searchBtn.disabled = busy;
    searchBtn.classList.toggle("is-busy", busy);
}

async function runSearch(loader: () => Promise<SearchResponse>): Promise<void> {
    const seq = ++requestSeq;
    setBusy(true);
    renderSkeletons(8);
    setStatus("检索中…");

    try {
        const data = await loader();
        if (seq !== requestSeq) {
            return;
        }
        renderResults(data);
        renderStatus(data);
    } catch (error) {
        if (seq !== requestSeq) {
            return;
        }
        setStatus("检索失败：" + describeError(error), true);
        renderEmpty("这次检索没有完成。确认后端已经启动，然后重试。");
    } finally {
        if (seq === requestSeq) {
            setBusy(false);
        }
    }
}

function clearQueryPreview(): void {
    if (previewObjectUrl !== null) {
        URL.revokeObjectURL(previewObjectUrl);
        previewObjectUrl = null;
    }
    if (queryPreview !== null) {
        queryPreview.hidden = true;
        queryPreview.removeAttribute("src");
    }
    if (dropzoneHint !== null) {
        dropzoneHint.textContent = "点击选择或拖拽图片到此处";
    }
}

function submitTextQuery(): void {
    const query = textInput.value.trim();
    if (query === "") {
        setStatus("先写一句描述，比如 a dog running on grass。", true);
        textInput.focus();
        return;
    }
    clearQueryPreview();
    void runSearch(() => searchByText(query));
}

searchBtn.addEventListener("click", submitTextQuery);

textInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        event.preventDefault();
        submitTextQuery();
    }
});

document.querySelectorAll<HTMLButtonElement>(".example").forEach((example) => {
    example.addEventListener("click", () => {
        const query = example.dataset.query;
        if (query === undefined || query === "") {
            return;
        }
        textInput.value = query;
        clearQueryPreview();
        void runSearch(() => searchByText(query));
    });
});

/* ---------- 图片检索 ---------- */

function shortenFileName(name: string, max: number): string {
    if (name.length <= max) {
        return name;
    }
    return name.slice(0, max - 1) + "…";
}

function handleImageFile(file: File): void {
    if (!file.type.startsWith("image/")) {
        setStatus("请选择图片文件（JPG 或 PNG）。", true);
        return;
    }
    if (previewObjectUrl !== null) {
        URL.revokeObjectURL(previewObjectUrl);
    }
    previewObjectUrl = URL.createObjectURL(file);
    if (queryPreview !== null) {
        queryPreview.src = previewObjectUrl;
        queryPreview.alt = file.name;
        queryPreview.hidden = false;
    }
    if (dropzoneHint !== null) {
        dropzoneHint.textContent = "已选择 " + shortenFileName(file.name, 26);
    }
    void runSearch(() => searchByImage(file)).then(() => {
        if (imageInput !== null) {
            imageInput.value = "";
        }
    });
}

if (imageInput !== null) {
    imageInput.addEventListener("change", () => {
        const file = imageInput.files?.[0];
        if (file === undefined) {
            return;
        }
        handleImageFile(file);
    });
}

if (dropzone !== null) {
    dropzone.addEventListener("dragenter", (event) => {
        event.preventDefault();
        dropzone.classList.add("is-drag");
    });
    dropzone.addEventListener("dragover", (event) => {
        event.preventDefault();
        dropzone.classList.add("is-drag");
    });
    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("is-drag");
    });
    dropzone.addEventListener("drop", (event) => {
        event.preventDefault();
        dropzone.classList.remove("is-drag");
        const file = event.dataTransfer?.files?.[0];
        if (file !== undefined) {
            handleImageFile(file);
        }
    });
}

/* ---------- 大图 ---------- */

function updateLightboxNav(): void {
    if (lightboxPrevBtn !== null) {
        lightboxPrevBtn.disabled = currentLightboxIndex <= 0;
    }
    if (lightboxNextBtn !== null) {
        lightboxNextBtn.disabled = currentLightboxIndex < 0 || currentLightboxIndex >= currentResults.length - 1;
    }
}

function renderLightboxItem(item: SearchResult): void {
    const captions = Array.isArray(item.captions) ? item.captions : [];
    const score = Number(item.score);

    if (lightboxImg !== null) {
        lightboxImg.src = item.url;
        lightboxImg.alt = captions.length > 0 ? captions[0] : item.filename;
    }
    if (lightboxCaption !== null) {
        lightboxCaption.innerHTML = "";

        const file = document.createElement("p");
        file.className = "lb-file";

        const left = document.createElement("span");
        left.textContent = item.filename;

        const right = document.createElement("span");
        const rankText = currentLightboxIndex >= 0 ? `#${currentLightboxIndex + 1} · ` : "";
        right.textContent = rankText + "相似度 " + (isFinite(score) ? score.toFixed(3) : "-");

        file.appendChild(left);
        file.appendChild(right);
        lightboxCaption.appendChild(file);

        if (captions.length > 0) {
            const list = document.createElement("ul");
            list.className = "lb-caps";
            for (const caption of captions) {
                const li = document.createElement("li");
                li.textContent = caption;
                list.appendChild(li);
            }
            lightboxCaption.appendChild(list);
        }
    }
    updateLightboxNav();
}

function openLightboxAt(index: number): void {
    if (lightbox === null) {
        throw new Error("找不到灯箱，检查id");
    }
    const item = currentResults[index];
    if (item === undefined) {
        return;
    }
    currentLightboxIndex = index;
    renderLightboxItem(item);
    lightbox.classList.add("show");
    lightbox.setAttribute("aria-hidden", "false");
}

function closeLightbox(): void {
    if (lightbox === null) {
        throw new Error("找不到灯箱，检查id");
    }
    currentLightboxIndex = -1;
    lightbox.classList.remove("show");
    lightbox.setAttribute("aria-hidden", "true");
}

function stepLightbox(delta: number): void {
    const nextIndex = currentLightboxIndex + delta;
    if (nextIndex < 0 || nextIndex >= currentResults.length) {
        return;
    }
    openLightboxAt(nextIndex);
}

lightbox?.addEventListener("click", (event) => {
    if (lightboxStage !== null && lightboxStage.contains(event.target as Node)) {
        return;
    }
    closeLightbox();
});

lightboxCloseBtn?.addEventListener("click", (event) => {
    event.stopPropagation();
    closeLightbox();
});

lightboxPrevBtn?.addEventListener("click", (event) => {
    event.stopPropagation();
    stepLightbox(-1);
});

lightboxNextBtn?.addEventListener("click", (event) => {
    event.stopPropagation();
    stepLightbox(1);
});

document.addEventListener("keydown", (event) => {
    if (lightbox === null || !lightbox.classList.contains("show")) {
        return;
    }
    if (event.key === "Escape") {
        closeLightbox();
    } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        stepLightbox(-1);
    } else if (event.key === "ArrowRight") {
        event.preventDefault();
        stepLightbox(1);
    }
});

/* ---------- 初始化 ---------- */

setStatus("");
renderEmpty("输入一句描述开始检索，比如 a dog running on grass。");
