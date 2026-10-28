# Sample data

Everything in this folder is fictional. Tallowmere Outfitters, an outdoor gear retailer,
was invented for this project, together with its people, stores, addresses, systems and
policies. Phone numbers use the 555-01xx range reserved for fiction, and email addresses
use the reserved `.example` domain.

| Folder | Contents |
|---|---|
| `docs/` | The 20 documents the pipeline indexes: 10 markdown files and 10 PDFs |
| `pdf-sources/` | The text the 10 PDFs are generated from. Not indexed. |
| `eval/` | The test set: `questions.csv`, 50 answerable questions and 5 that no document answers, and `facts.csv`, the facts each answer relies on |

## The documents

Topics overlap the way they do in a real company, but each fact appears in only one
document, so every answerable question has a single correct source.

| Document | Owner | Created | Notes |
|---|---|---|---|
| `benefits-overview.pdf` | People Team | 2026-01-05 | Bookmarks. The "401(k) plan" bookmark is printed as "Saving for retirement" |
| `code-of-conduct.pdf` | Legal and Ethics | 2021-03-15 | No bookmarks |
| `company-overview-faq.md` | Communications | 2026-03-02 | |
| `data-retention-policy.pdf` | Legal and Ethics | 2022-04-25 | No bookmarks |
| `employee-handbook-time-off.md` | People Team | 2022-01-10 | Nested headings |
| `gear-rental-faq.md` | Store Operations | 2021-05-03 | |
| `it-helpdesk-faq.md` | IT and Security | 2024-06-11 | |
| `it-security-policy.pdf` | IT and Security | 2025-02-03 | Bookmarks |
| `loyalty-program-terms.pdf` | Marketing | 2023-11-01 | Bookmarks |
| `onboarding-checklist.md` | People Team | 2023-03-06 | |
| `parental-leave-policy.pdf` | People Team | 2025-08-18 | Bookmarks |
| `performance-review-process.md` | People Team | 2024-01-15 | |
| `price-match-policy.md` | Merchandising | 2025-04-22 | |
| `remote-work-policy.md` | People Team | 2024-09-30 | |
| `returns-and-refunds-policy.pdf` | Customer Experience | 2024-02-12 | Bookmarks |
| `shipping-faq.md` | Customer Experience | 2023-06-19 | |
| `store-opening-and-closing-procedures.md` | Store Operations | 2022-11-14 | Nested headings |
| `travel-and-expense-policy.pdf` | Finance | 2023-07-24 | Bookmarks. A section continues across the page break |
| `warranty-and-repairs.pdf` | Product Quality | 2022-09-07 | No bookmarks |
| `workplace-safety-and-incident-reporting.pdf` | Operations Safety | 2023-10-09 | Bookmarks |

The created date is stored where a loader would find it in real documents: the
`/CreationDate` field of each PDF, and a `created:` line in each markdown file's front
matter.

## Changing a PDF

Edit its source in `pdf-sources/`, then run `make pdfs`. The build is deterministic, and a
test fails if any PDF in `docs/` no longer matches its source. Sources are plain ASCII,
because the PDFs use the standard Helvetica font.

## The test set

`eval/questions.csv` is a standard CSV file, UTF-8, with this header:

```csv
id,question,expected_doc,expected_answer
```

| Column | Required | Meaning |
|---|---|---|
| `id` | Yes | Unique id, such as `q01` |
| `question` | Yes | The question, phrased the way an employee or customer would ask it |
| `expected_doc` | For answerable questions | Exact file name of the one document that contains the answer, such as `shipping-faq.md` |
| `expected_answer` | No | A short reference answer, used only when grading answers |

Leave `expected_doc` empty for a question that no document answers. Those questions are
left out of the 50 and out of the retrieval hit rate; they test whether the chat model
says it doesn't know. Put a field in double quotes if it contains a comma.

Two rows from the file, one of each kind:

```csv
q39,How much do I have to spend for free delivery?,shipping-faq.md,"$75 or more, after discounts and before tax, for free standard shipping."
u02,What is the password for the guest Wi-Fi?,,
```

The questions were written together with the documents rather than independently;
`docs/decisions.md` (decision 31) explains what that means for the results.

## The fact list

`eval/facts.csv` lists, for every answerable question, the facts its expected answer
relies on, one row per fact, worded exactly as the expected document states them:

```csv
id,fact
q16,$200 for kayaks and paddleboards
q16,released when the gear comes back in good condition
q16,Debit cards and cash are not accepted for the deposit
```

`rag eval` uses it for answer coverage: whether the top retrieved chunks contain every
fact of the answer (decision 39). Every answerable question needs at least one fact, and
the questions that no document answers have none. Like the questions, the facts are
frozen: neither file is changed after results have been seen.
