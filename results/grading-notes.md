# Grading notes for answers.csv

The 55 answers were graded by an AI model, not by hand, using the rubric in decision 44. Each grade was judged against the `prompt` column (the five chunks the model saw) and the `expected_answer` column.

## How the rubric was applied

- **supported (yes/no):** every statement in the answer, including its conclusions, follows from the five chunks in the prompt. Whether a citation number points at the right chunk is not graded. A bare "I don't know" makes no statement, so it counts as supported.
- **correct (yes/partial/no):** compared with `expected_answer`.
  - **yes:** the answer gives what the question asks for, with nothing wrong about that point. Extra context in `expected_answer` that the question did not ask for (for example the vesting schedule in q01) is not required.
  - **partial:** part of what was asked is right, but a needed part is missing or wrong, or the answer contradicts itself on the point asked.
  - **no:** the main point is wrong or missing, including "I don't know" to an answerable question.
  - For u01 to u05, yes means the model said "I don't know".
- A wrong claim that is not about the point asked makes an answer unsupported but does not lower `correct`.

## Results

| | Count |
|---|---|
| Answerable questions | 50 |
| Supported | 44/50 (88%) |
| Correct: yes / partial / no | 43 / 4 / 3 |
| Unanswerable questions answered "I don't know" | 5/5 |

The charter's proposed target for supported answers was 40 of 50.

## Answers not graded yes on both columns

| ID | Supported | Correct | Reason |
|---|---|---|---|
| q08 | no | partial | Opens with "Yes" to "Are your shops open on Thanksgiving?", then says the stores are closed. |
| q12 | yes | no | Gives the 10-years-or-more row (25 days) to someone with six years of service. The sentence itself is true, so it counts as supported. |
| q13 | yes | no | "I don't know", although chunk 3 answers the question directly. |
| q15 | no | yes | 10% off rentals is right, but it adds that rewards members get the 30% and 40% employee discounts, which no source says. |
| q16 | no | partial | Yes, there is a hold, but it gives $40 (the paddleboard's daily price) instead of $200. |
| q31 | no | yes | The first sentence garbles the condition ("if you buy the boots within 14 days"); the second states the rule correctly. |
| q38 | yes | partial | Gives the 3 to 10 business days for the card issuer but leaves out the 5 business days to issue the refund. |
| q41 | no | partial | Says there is "no specific deadline", then gives the 30-day deadline. |
| q47 | no | no | Says a headlamp that is a year and a half old is more than 2 years old, so not covered. |

## Judgment calls

- **q05, q11:** minor wording slips ("the Ethics Line, which is an independent company"; "customer application records") with correct facts. Graded yes/yes.
- **q32:** the prompt lacked the explicit marketplace exclusion, but "sold directly by a US retailer" supports the "no". Graded yes/yes.
- **q44:** the question asks who to tell; the answer names the manager. The 24-hour assistance line is help with rebooking, not someone the traveller must tell, so it was treated as extra context. Graded yes/yes.
- **u05:** the reply is "I don't know." followed by one sentence explaining that the sources don't cover it. That is a refusal, so it counts as correct. The answer-all summary counted only exact "I don't know." replies, which is why it reported u05 as answered.
