# Release checklist (maintainer note)

## First release -> DOI

1. **Zenodo, in this order.** Go to https://zenodo.org/account/settings/github/ and click
   **Sync now** first. Then find `311cleiton/vdW-GNN-SAFT` and switch it **ON**.
   (Sync-before-toggle matters: Zenodo caches the repository list by GitHub's internal
   repo id, and a stale cache is what produces "Request failed with status code: 403".)

2. Publish the release:

       gh release create v1.0.0 --title "vdW-GNN-SAFT v1.0.0" --notes "First public release."

3. Wait about a minute, reload the Zenodo GitHub page, open the new record, and copy the
   **concept DOI** - the one labelled "Cite all versions". Not the version DOI: the concept
   DOI always resolves to the newest release.

4. Add the DOI by hand (two small edits):

   - `CITATION.cff` - directly under the `date-released:` line, add:

         doi: "10.5281/zenodo.XXXXXXX"

   - `README.md` - directly under the title line, add the badge:

         [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

5. Commit, push, and (optional but tidy) cut v1.0.1 so the archived snapshot itself
   carries the DOI:

       git commit -am "Add Zenodo DOI"
       git push
       gh release create v1.0.1 --title "vdW-GNN-SAFT v1.0.1" --notes "Add Zenodo DOI."

6. Paste the concept DOI into the manuscript's Data Availability statement.

## Later releases

Tag a new version (`gh release create v1.x.y ...`); Zenodo archives it automatically under
the same concept DOI. Update `version:` and `date-released:` in `CITATION.cff` when you do.
