// Answers can end at the next chapter/continuation or at the end of the article.
const answerSection = /\n### 参考答案\n([\s\S]*?)(?=\n(?:## |下一篇(?:会把|把))|$(?![\s\S]))/;
export const withoutReferenceAnswers = source => source.replace(answerSection,'');
export const referenceAnswers = source => source.match(answerSection)?.[1] ?? '';
