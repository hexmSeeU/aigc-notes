// Exclude only the newly added answer subsection when checking immutable source content.
export const withoutReferenceAnswers = source => source.replace(/\n### 参考答案\n[\s\S]*?(?=\n(?:## |下一篇(?:会把|把)))/g,'');
export const referenceAnswers = source => source.match(/\n### 参考答案\n([\s\S]*?)(?=\n(?:## |下一篇(?:会把|把)))/)?.[1] ?? '';
