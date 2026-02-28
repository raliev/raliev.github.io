---
layout: post
title: "Gemini API offers significant architectural and functional advantages over GPT-4 API, particularly when handling complex, multi-page scientific document"
date: 2026-02-27
tags:
  - Retrieval-Augmented Generation (RAG)
  - Multimodal Search (Text, Image, Audio)
  - Algorithms & Models
  - Information Retrieval (IR)
---

# Gemini API offers significant architectural and functional advantages over GPT-4 API, particularly when handling complex, multi-page scientific documents

The core task was to take a large volume PDF, load it once, and ask several questions across separate, independent sessions. The goal was to avoid re-uploading the file and paying for the same input tokens every time by utilizing existing API caching mechanisms. Additionally, it was critical that the LLM processes both text and embedded images accurately from these large documents. 

To test this, I analyzed a 39-page research paper ([arXiv:2310.01783](https://arxiv.org/abs/2310.01783)) using both Gemini and OpenAI GPT-4. The results highlight fundamental differences in how these systems "see", "remember", and handle pricing for large documents.

Here are the key takeaways from the comparison.

### 1. Native Multi-Modal Understanding vs. Textual RAG

Scientific papers are not just text; they are a synthesis of formulas, charts, and diagrams. When working with large volume PDFs, it is essential that the model effortlessly understands both text and images together.

OpenAI’s Assistants API handles PDFs primarily through File Search (a Retrieval-Augmented Generation, or RAG, approach). GPT-4o extracts the text and stores it in a vector database. To get the API to "see" a diagram, you have to manually crop it, convert it to an image, and send it via the Vision API. For large documents, this workaround is slow, expensive, and breaks context. *(Note: ChatGPT functions differently as an agentic system, but the API relies on this extraction method).*

**Gemini’s Advantage:** While the exact internal mechanics of how Gemini processes PDFs are not exhaustively documented, available documentation indicates it still parses the PDF into text and images within the pipeline, feeding both directly into its multimodal model. The true advantage is that this mechanism is built natively into Gemini and operates significantly faster than the manual, multi-step image cropping and separate database extraction required for the OpenAI GPT-4 API. This integrated, out-of-the-box multimodal approach makes Gemini exceptionally efficient and well-suited for processing large volume PDFs.

### 2. Visual Accuracy

To test visual reasoning, I asked both models to analyze Figure 3 in the document—a complex log-frequency ratio diagram.

![The diagram about which I asked the LLMs providing the PDF containing this diagram at the page 17](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdr5Cdzh217dPnNJGNBw7MNG3XfYScxLKUeXwqqGHYx7c3_vzSsnJXxp06UOMO9yh8xBZ56WakPWSnKko4urEoXPRDj7WL_4VpUosbDPADbuyF4STdMcJ3P221m74JsHsqf7UXGr1s9s84y0QYpvXOykgUrbw?key=1miJacsJFi0g4LKmYfBfaA)

* **Gemini** correctly identified the specific color coding of the diagram ("The primary colors used are shades of blue, purple, and red (specifically a reddish-pink or salmon color)... Blue/Deep Purple is used for aspects more frequently mentioned by human reviewers... Red/Pink is used for aspects more frequently mentioned by GPT-4."). This proves it is performing true visual reasoning on the document's content.
* **OpenAI GPT-4o**, despite having access to the file via its Assistant storage, provided a generic response about "shades of blue and red." It guessed standard chart colors based on the parsed text, failing to actually "see" the diagram.

### 3. Context Caching vs. Vector Stores

The way these models maintain "memory" and handle pricing for ongoing, separate sessions is fundamentally different. 

* **OpenAI (Assistants API):** To maintain a session, you must upload files to a Vector Store, manage a Thread ID, and manually delete files later to avoid storage costs. It is a "search-and-retrieve" architecture. The model only sees snippets of text relevant to your question, often missing the broader context of the paper. When querying the full length of a document directly without Vector stores, you end up paying full price for the input tokens every single time.
* **Gemini (Context Caching):** Gemini allows you to upload a file and create a Context Cache. This keeps the large document in "warm" memory. You can return a day later, in a completely separate session, ask a follow-up question, and the model immediately responds with the full context already loaded. Because of its specific pricing mechanisms, you pay significantly less for "cached tokens" compared to passing the document from scratch each time. There is no repetitive token billing for the full document, no "indexing" delay, and no "search" phase—it is a true, cost-effective persistent session.

### 4. Workflow Integrity

To mimic Gemini’s native performance with OpenAI, you would have to convert all 39 pages into 39 high-resolution images and send them in a single session. This often exceeds token limits and is remarkably inefficient. 

With Gemini, you simply attach the PDF. The model handles the text, the complex mathematical notations, and the logical flow of the paper's imagery in one go. 

While ChatGPT is excellent at processing extracted text, between the two tested models, **Gemini was the only API model capable of natively performing a "visual" peer review.** It understands not just what the researchers *wrote*, but what they *plotted*. 

### 5. Other Vendors (Claude and others)

It is worth noting that other options exist in the market, such as Anthropic's Claude 3.5 models. According to their documentation, Claude also offers [native PDF support featuring visual understanding](https://docs.anthropic.com/en/docs/build-with-claude/pdf-support) and [Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) mechanisms which similarly reduce latency and token costs for large contexts across separate queries. 

*Please note: No direct testing was conducted on Claude for this specific workflow, and these capabilities are mentioned based purely on their official documentation.*

---

### The Proof: Testing the Models with Edited Diagrams

To truly test visual reasoning, an experiment was conducted using LibreOffice Draw. Two separate PDFs were created from the original document:
1. A PDF with the original diagram (Figure 3) intact.
2. A PDF where the data points (the circles) in the diagram were manually deleted.

For the sake of experiment purity, both versions were saved directly from LibreOffice (so the underlying PDF layout remained identical, just with the content removed in the first). The primary goal was to check if the LLMs would actually notice that the diagram was empty, even though the surrounding text implies the data is present.

*(Note: The `gpt-4o` model was used for OpenAI testing because, as of February 2026, it is the latest model that supports the Assistants API, which is required for uploading a file to the server. A test was also conducted on `gpt-5.2`, but no difference in behavior was noticed compared to `gpt-4o`. The alternative of manually converting the PDF into text and images before sending was not even considered, because the very premise of this evaluation expects that functionality to be handled natively by the service.)*

#### OpenAI GPT-4o on PDF with edited (empty) picture:

> **Response:** The log frequency ratio diagram in this paper presents a comparison between comments made by a language model (LLM) and human reviewers across several aspects of academic feedback. This diagram is used to demonstrate how feedback emphasis differs between LLMs and human reviewers. Specifically, it includes the following aspects: novelty, ablation experiments, missing citations, theoretical soundness, comparison to previous studies, reproducibility, algorithm efficiency, ethical aspects, clarity and presentation, additional experiments on more datasets, and implications of the research.
>
> Key insights from the diagram include the finding that LLM feedback emphasizes certain aspects more than human feedback. For example, LLMs comment on the implications of research much more frequently (7.27 times) than human reviewers. Conversely, LLMs are noted to be significantly less likely (10.69 times less) to comment on novelty compared to human reviewers. Additionally, while both LLMs and humans suggest additional experiments, humans are significantly more likely (6.71 times more) to request more ablation experiments, whereas LLMs tend to request experiments on more datasets more frequently (2.19 times more) than human reviewers【4:1†art1.pdf】

#### OpenAI GPT-4o on PDF with unedited picture:

> **Response:** The log frequency ratio diagram in the document visualizes the frequency at which different aspects of feedback are addressed by GPT-4 compared to human reviewers. The X-axis represents the categories of feedback, such as Novelty, Theoretical Soundness, and Clarity and Presentation. The Y-axis shows the log frequency ratio between comments from GPT-4 and human reviewers. In this diagram, a ratio above zero indicates that GPT-4 comments on that aspect more frequently than humans, while a ratio below zero suggests that humans comment more on that aspect than GPT-4.
>
> Notably, GPT-4 emphasizes research implications significantly more often than human reviewers (with a log frequency ratio of 7.27), but it comments on novelty considerably less frequently (10.69 times less) compared to humans. Additionally, while both GPT-4 and human reviewers frequently suggest additional experiments, their focuses differ; humans are more likely to request ablation experiments, whereas GPT-4 tends to suggest experiments on more datasets. The size of circles in the diagram indicates the prevalence of each aspect in human feedback【4:0†art2.pdf】

#### Gemini 3 Preview on PDF with edited (empty) picture:

> **--- Phase 1: Initial Diagram Analysis ---**
> 
> **User:** Please analyze the log frequency ratio diagram in this PDF.
> 
> **Gemini:** Based on the diagram in Figure 3 on page 17 of the PDF, here is a visual and accurate analysis:
> 
> ### **Overview**
> The diagram is a horizontal dot plot with error bars titled **"Figure 3. LLM based feedback emphasizes certain aspects more than humans."** It illustrates the relative frequency of 11 different scientific feedback aspects mentioned by GPT-4 compared to human reviewers.
> 
> ### **Axes and Scale**
> *   **X-axis (Bottom):** Represents the **"log Frequency Ratio (GPT-4/Human)"** on a scale from -1 to 1. 
>     *   **Positive values (to the right of 0):** Indicate that the aspect is mentioned more frequently by GPT-4.
>     *   **Negative values (to the left of 0):** Indicate that the aspect is mentioned more frequently by human reviewers.
> *   **Y-axis (Left):** Lists 11 categories of feedback aspects, ordered from those most emphasized by humans at the bottom to those most emphasized by GPT-4 at the top.
> 
> ### **Categorical Breakdown**
> The data points (represented by circles with horizontal error bars) show the following distribution:
> 
> **Aspects favored by GPT-4 (Positive Ratio):**
> *   **Implications of the Research:** This is the most significant outlier toward GPT-4. According to the caption, GPT-4 mentions this **7.27 times more frequently** than humans.
> *   **Add experiments on more datasets:** This is also prominently on the GPT-4 side. The text notes GPT-4 is **2.19 times more likely** to make this request.
> *   **Ethical Aspects:** Shows a slight leaning toward being mentioned more by GPT-4.
> 
> **Aspects favored by Human Reviewers (Negative Ratio):**
> *   **Novelty:** This is the most significant outlier toward human reviewers. The caption states LLM is **10.69 times less likely** to comment on novelty than humans.
> *   **Add ablations experiments:** Humans are **6.71 times more likely** than the LLM to request these.
> *   **Missing Citations:** leans noticeably toward human reviewers.
> *   **Comparison to Previous Studies:** leans toward human reviewers.
> 
> **Aspects with relatively balanced frequency (Near 0):**
> *   **Theoretical Soundness**, **Reproducibility**, **Algorithm Efficiency**, and **Clarity and Presentation** are clustered closer to the center line, indicating more similar frequencies of mention between humans and the LLM.
> 
> ### **Key takeaway**
> The diagram visually confirms the paper's finding that while LLMs can identify similar general areas of concern, they have a distinct "style" of feedback, focusing heavily on research implications and dataset breadth while struggling to identify issues of novelty or the need for specific ablation studies compared to human experts.
> 
> **--- Phase 2: Follow-up Question ---**
> 
> **User:** what are the primary colors used in the log frequency ratio diagram?
> 
> **Gemini:** Based on the "log Frequency Ratio (GPT-4/Human)" diagram (Figure 3 on page 17), the primary colors used for the data points (represented as bubbles of varying sizes) are **blue, green, orange, purple, red, and yellow**.
> 
> Specifically, the diagram uses a diverse categorical color palette to distinguish between the 11 different aspects of research feedback (such as "Novelty," "Missing Citations," and "Theoretical Soundness"). The background of the diagram is white, and the axes and labels are in black.

#### Gemini 3 Preview on PDF with unedited picture:

> **--- Phase 1: Initial Diagram Analysis ---**
> 
> **User:** Please analyze the log frequency ratio diagram in this PDF.
> 
> **Gemini:** Based on Figure 3 on page 17, titled **"LLM based feedback emphasizes certain aspects more than humans,"** here is a visual and accurate analysis of the log frequency ratio diagram:
> 
> ### **Overview**
> The diagram is a horizontal dot plot (or bubble chart) that compares how frequently 11 different aspects of research paper feedback are raised by GPT-4 versus human reviewers.
> 
> ### **Axes and Scales**
> *   **X-axis:** Labeled **"log Frequency Ratio (GPT-4/Human)"**. It ranges from **-1** on the left to **1** on the right, with a dashed vertical line at **0**.
>     *   **Positive values (>0):** Indicate the aspect is mentioned more frequently by GPT-4.
>     *   **Negative values (<0):** Indicate the aspect is mentioned more frequently by human reviewers.
> *   **Y-axis:** Lists **11 categories of feedback**, ordered from most frequent in LLM reviews at the top to most frequent in human reviews at the bottom.
> 
> ### **Key Visual Indicators**
> *   **Bubble Color:** Features a gradient from **red/pink** (top right, favoring LLM) to **purple** (center) to **blue** (bottom left, favoring humans).
> *   **Bubble Size:** Indicates the **prevalence of each aspect in human feedback**. Larger circles mean that category appears more often in traditional peer reviews.
> 
> ### **Data Analysis by Category**
> 1.  **LLM-Leaning Aspects (Right side):**
>     *   **Implications of the Research:** Has the highest positive ratio (~0.8). It is a small red dot, meaning it's much more common in LLM feedback but relatively rare in human feedback.
>     *   **Add experiments on more datasets:** A pink dot located at ~0.3.
>     *   **Clarity and Presentation:** This is the largest bubble, located slightly to the right of zero. This indicates it is the most common feedback point overall and is raised slightly more often by the LLM.
> 
> 2.  **Neutral Aspects (Center):**
>     *   **Ethical Aspects**, **Algorithm Efficiency**, and **Reproducibility** are clustered near the center line (0), meaning they are mentioned at roughly similar rates by both sources. These categories have small bubble sizes.
> 
> 3.  **Human-Leaning Aspects (Left side):**
>     *   **Comparison to Previous Studies:** Slightly to the left (~-0.1) with a medium-sized bubble.
>     *   **Theoretical Soundness:** A medium-sized blue bubble at ~-0.4.
>     *   **Missing Citations:** A small blue dot at ~-0.6.
>     *   **Add ablations experiments:** A very small blue dot at ~-0.8.
>     *   **Novelty:** Has the lowest ratio (~ -1.0) with a medium-sized blue bubble, indicating that humans are significantly more likely to comment on a paper's novelty than the LLM is.
> 
> ### **Conclusion of the Diagram**
> The diagram visually demonstrates that while LLMs excel at pointing out broad implications and presentation issues, they struggle to match human reviewers in identifying technical shortcomings (like missing ablations), identifying missing citations, or judging the fundamental novelty and theoretical soundness of a research paper.
> 
> **--- Phase 2: Follow-up Question ---**
> 
> **User:** what are the primary colors used in the log frequency ratio diagram?
> 
> **Gemini:** In the log frequency ratio diagram (Figure 3 on page 17), the primary colors used for the data points are:
> 
> *   **Shades of Red and Pink:** Used for aspects that the LLM (GPT-4) mentions more frequently than humans (points to the right of the center line), such as "Implications of the Research" and "Add experiments on more datasets."
> *   **Shades of Blue and Purple:** Used for aspects that human reviewers mention more frequently than the LLM (points to the left of the center line), such as "Novelty," "Theoretical Soundness," and "Missing Citations."