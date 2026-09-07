import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function MarkdownAnswer({ answer }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ node, ...props }) => <a {...props} target="_blank" rel="noreferrer" />,
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
