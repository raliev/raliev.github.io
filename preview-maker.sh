#!/bin/bash
#
# build-sample.sh
# Скрипт для создания "демонстрационного" PDF с полным содержанием
# и N первыми страницами каждой главы.
#
set -e # Выход при любой ошибке

# --- 1. Конфигурация ---
N_SAMPLE_PAGES=5 # СКОЛЬКО СТРАНИЦ БРАТЬ ИЗ КАЖДОЙ ГЛАВЫ
BOOK_URL="https://testmysearch.com/my-books.html"

# Имена файлов (используем полные пути для надежности)
SCRIPT_DIR=$(pwd)
COMBINED_MD_FILE="$HOME/combined-ecomsearch.markdown"
FULL_TEX_FILE="$HOME/ecomsearch-full.tex"
FULL_PDF_NAME="ecomsearch-full" # Имя без .pdf
FULL_AUX_FILE="${FULL_PDF_NAME}.aux"
FULL_PDF_FILE="${FULL_PDF_NAME}.pdf"


# --- 2. Сборка полного PDF (Шаг 1 из 2 в оригинальном скрипте) ---

echo "--- Шаг 1: Сборка Markdown в единый файл ---"
cat docs/_posts/2025-10-10-Designing-Ecommerce-Search-{1,2,3,3a,4,4a,4b,4c,5,6,7,7a,8}.markdown > ${COMBINED_MD_FILE}
echo "OK: ${COMBINED_MD_FILE}"

echo "--- Шаг 2: Конвертация MD в TEX ---"
python3 md-to-latex-convertor7-ecomsearch.py ${COMBINED_MD_FILE} -o ${FULL_TEX_FILE}
echo "OK: ${FULL_TEX_FILE}"

echo "--- Шаг 3: Полная компиляция PDF (может занять время) ---"
# Мы должны перейти в директорию, где лежит .tex, чтобы pdflatex нашел изображения
TEX_DIR=$(dirname ${FULL_TEX_FILE})
TEX_BASENAME=$(basename ${FULL_TEX_FILE})

# Компилируем дважды для разрешения ссылок и содержания
(cd ${TEX_DIR} && pdflatex -shell-escape -interaction=nonstopmode ${TEX_BASENAME})
(cd ${TEX_DIR} && pdflatex -shell-escape -interaction=nonstopmode ${TEX_BASENAME})

# Перемещаем .pdf и .aux в текущую директорию для удобства
mv "${TEX_DIR}/${FULL_PDF_NAME}.pdf" "${SCRIPT_DIR}/${FULL_PDF_FILE}"
mv "${TEX_DIR}/${FULL_PDF_NAME}.aux" "${SCRIPT_DIR}/${FULL_AUX_FILE}"
echo "OK: Создан ${FULL_PDF_FILE} и ${FULL_AUX_FILE}"


# --- 3. Создание PDF-заглушки ---

echo "--- Шаг 4: Создание PDF-страницы (заглушки) ---"
PLACEHOLDER_TEX="placeholder.tex"
PLACEHOLDER_PDF="placeholder.pdf"

# Используем тот же размер бумаги, что и в вашем конвертере
cat << EOF > ${PLACEHOLDER_TEX}
\\documentclass[11pt]{scrbook}
\\usepackage[
    paperwidth=7in,
paperheight=10in,
inner=0.85in,
outer=0.65in,
top=0.75in,
bottom=0.5in
]{geometry}
\\usepackage{hyperref}
\\usepackage[utf8]{inputenc}
\\usepackage[T1]{fontenc}
\\hypersetup{colorlinks=true, urlcolor=blue}
\\begin{document}
\\pagestyle{empty}
\\vspace*{8cm}
\\begin{center}
\\Huge This is a demonstration PDF.
\\[3cm]
\\Large The full book is available for purchase:
    \\[1cm]
\\Large \\url{${BOOK_URL}}
\\end{center}
\\end{document}
EOF

pdflatex -interaction=nonstopmode ${PLACEHOLDER_TEX} > /dev/null
echo "OK: Создан ${PLACEHOLDER_PDF}"


# --- 4. Анализ и Сборка Демо-PDF ---

echo "--- Шаг 5: Анализ ${FULL_AUX_FILE} для получения страниц глав ---"

# Ваш Python-скрипт использует \chapter для заголовков H1 (level 1).
# В .aux файле это выглядит как: \newlabel{sec:slug}{{1}{СТРАНИЦА}}
# Мы извлекаем все номера СТРАНИЦ для всех записей уровня 1.
CHAPTER_START_PAGES=()
while read -r line; do
PAGE=$(echo "$line" | sed -n 's/.*}{\([0-9]*\)}$/\1/p')
CHAPTER_START_PAGES+=(${PAGE})
done < <(grep "newlabel{sec:.*}{{1}{" ${FULL_AUX_FILE})

# Получаем общее кол-во страниц
TOTAL_PAGES=$(pdfinfo ${FULL_PDF_FILE} | grep "Pages:" | awk '{print $2}')

# Добавляем "виртуальную" главу в конце, чтобы знать, где кончается последняя
CHAPTER_START_PAGES+=($((TOTAL_PAGES + 1)))

echo "Найдено ${#CHAPTER_START_PAGES[@]} глав (включая маркер конца)"
echo "Главы начинаются на страницах: ${CHAPTER_START_PAGES[*]}"

# Содержание заканчивается на странице ПЕРЕД началом первой главы
FIRST_CHAPTER_START_PAGE=${CHAPTER_START_PAGES[0]}
TOC_END_PAGE=$((FIRST_CHAPTER_START_PAGE - 1))

echo "Содержание (TOC) и вступление идут со страницы 1 по ${TOC_END_PAGE}."

echo "--- Шаг 6: Генерация 'обертки' (sample-wrapper.tex) ---"
SAMPLE_TEX_FILE="sample-wrapper.tex"
FINAL_SAMPLE_PDF_NAME="ecomsearch-sample"

# Используем `cat` (Heredoc) для создания .tex файла
cat << EOF > ${SAMPLE_TEX_FILE}
\\documentclass{scrbook}
\\usepackage[final]{pdfpages}
\\begin{document}
\\pagestyle{empty}

% --- Часть 1: Включаем всё до первой главы (Содержание и т.д.) ---
\\includepdf[pages={1-${TOC_END_PAGE}}, final]{${FULL_PDF_FILE}}

% --- Часть 2: Циклом проходим по главам и вставляем части ---
EOF

# Проходим по всем главам, кроме последней (виртуальной)
for i in $(seq 0 $((${#CHAPTER_START_PAGES[@]} - 2)))
do
CH_START=${CHAPTER_START_PAGES[$i]}
NEXT_CH_START=${CHAPTER_START_PAGES[$i+1]}

# Последняя страница *этой* главы - это страница *перед* началом *следующей*
CH_LAST_PAGE=$((NEXT_CH_START - 1))

# Рассчитываем последнюю страницу для демо-версии
SAMPLE_END_PAGE=$((CH_START + N_SAMPLE_PAGES - 1))

ADD_PLACEHOLDER=false

# Проверяем, не выходим ли мы за пределы главы
if [ ${SAMPLE_END_PAGE} -gt ${CH_LAST_PAGE} ]; then
# Глава короче N страниц, берем ее целиком
PAGE_RANGE="${CH_START}-${CH_LAST_PAGE}"
else
# Глава длинная, урезаем ее и ставим флаг на добавление заглушки
PAGE_RANGE="${CH_START}-${SAMPLE_END_PAGE}"
ADD_PLACEHOLDER=true
fi

# Добавляем команды в .tex файл
echo "  % Включаем Главу $((i+1)): страницы ${PAGE_RANGE}" >> ${SAMPLE_TEX_FILE}
echo "  \\includepdf[pages={${PAGE_RANGE}}, final]{${FULL_PDF_FILE}}" >> ${SAMPLE_TEX_FILE}

if ${ADD_PLACEHOLDER}; then
echo "  % Добавляем заглушку после Главы $((i+1))" >> ${SAMPLE_TEX_FILE}
echo "  \\includepdf[pages={1}, final]{${PLACEHOLDER_PDF}}" >> ${SAMPLE_TEX_FILE}
fi
done

# Заканчиваем .tex файл
echo "\\end{document}" >> ${SAMPLE_TEX_FILE}

echo "OK: ${SAMPLE_TEX_FILE} создан."

echo "--- Шаг 7: Компиляция финального Демо-PDF ---"
pdflatex -interaction=nonstopmode ${SAMPLE_TEX_FILE}
mv "sample-wrapper.pdf" "${FINAL_SAMPLE_PDF_NAME}.pdf"

echo "--- ГОТОВО! ---"
echo "Полная версия: ${FULL_PDF_FILE}"
echo "Демо-версия: ${FINAL_SAMPLE_PDF_NAME}.pdf"

# --- 8. Очистка ---
rm -f placeholder.tex placeholder.log placeholder.aux placeholder.pdf
rm -f sample-wrapper.tex sample-wrapper.log sample-wrapper.aux
rm -f ${FULL_AUX_FILE} *.log

# Открываем результат
open "${FINAL_SAMPLE_PDF_NAME}.pdf"