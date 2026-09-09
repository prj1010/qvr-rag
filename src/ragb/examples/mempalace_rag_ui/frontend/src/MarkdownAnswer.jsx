import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function safeHref(href) {
  if (!href || typeof href !== "string") return undefined;
  return /^(https?:|mailto:|#)/i.test(href) ? href : undefined;
}

export default function MarkdownAnswer({ answer }) {
  if (!answer) return null;
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ node, href, ...props }) => (
          <a {...props} href={safeHref(href)} target="_blank" rel="noreferrer noopener" />
        ),
        table: ({ node, ...props }) => <div className="markdown-table-wrap"><table {...props} /></div>,
        pre: ({ node, ...props }) => <pre className="markdown-code-block" {...props} />,
        code: ({ node, inline, className, ...props }) => (
          <code className={inline ? "markdown-inline-code" : className || ""} {...props} />
        ),
      }}
    >
      {answer}
    </ReactMarkdown>
  );
}
