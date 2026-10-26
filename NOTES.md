# Notes

## How to run

1. Put your Cloudflare account ID and Workers AI API token in `cloud-flare.local.sh`:
   run `python classify.py --setup`, or copy `cloud-flare.sh` to `cloud-flare.local.sh`
   and edit it. `.gitignore` keeps this file out of git.
2. Run the program (Python 3.8+, standard library only, nothing to install):

   ```
   python classify.py --check      # sends one comment to each model to test the credentials
   python classify.py              # classifies all 100 comments with clef and clef-flash
   python classify.py --model clef # one model only
   python classify.py --refresh    # ignore saved results and call the API again
   python classify.py --html-only  # rebuild results.html from saved results
   python classify.py --doctor     # checks results, files, git, and that the token isn't committed
   ```

   It sends 4 requests at a time, retries rate limits and server errors, stops at the first
   error that a retry can't fix, and saves every answer received so far.

3. Open `results.html` in a browser.

Each model's answers are saved to `results-clef.json` and `results-clef-flash.json` as the
program runs, so a second run doesn't call the API again unless you pass `--refresh`.

`results.html` shows every comment with the label from `clef` (the larger model), the
probability of each label, and its confidence. Next to it is the label from `clef-flash`.
You can filter by label, by low confidence (below 0.5), or by comments where the two
models disagree, and sort with the least confident first. The page also has a review
column for choosing your own label and writing a reason, which it can export as
`MISTAKES.md`.

## Tools and models

- **Clef** and **Clef-flash** (`@cf/cloudflare/clef`, `@cf/cloudflare/clef-flash`) on
  Cloudflare Workers AI for the classification.
- **Claude** (Claude app) to write `classify.py`, including the instructions and criteria
  for acceptable, toxic and spam and the review page, and to explain the Clef API and the
  git steps.
- `MISTAKES.md`: I read all 100 comments and Clef's labels myself and chose every label and
  reason without AI. The review page only collected my answers and formatted the file.

## What I felt about the task

[Write 3 to 5 lines yourself: what was easy, what was hard, what surprised you about
Clef's mistakes, and what you would change in the criteria to reduce them.]
