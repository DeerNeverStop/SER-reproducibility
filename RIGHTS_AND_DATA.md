# Rights, data access and attribution

This repository provides public access to research code, manuscripts, configurations and reported results. Publication of this snapshot is not a blanket MIT, Apache or Creative Commons grant. No repository-wide reuse license has yet been selected for the original work. Third-party material retains its own terms; contact the repository owner if your intended reuse requires an additional grant.

## Datasets and pretrained models

Raw audio, voice embeddings, feature caches, pretrained model weights and fine-tuned checkpoints are not included in the source repository. Obtain them from their original providers and comply with the applicable terms. File lists, corpus-native IDs, split assignments and hashes are research metadata, not newly collected audio or a new speech dataset.

| Resource | Original source and rights information |
|---|---|
| CREMA-D | [Official repository](https://github.com/CheyneyComputerScience/CREMA-D): the database is distributed under ODbL 1.0 and individual contents under DbCL 1.0. Preserve the applicable attribution and database terms for derived metadata. |
| RAVDESS | [Original dataset record](https://doi.org/10.5281/zenodo.1188976); the historical acquisition record reports CC BY-NC-SA 4.0. See the preserved [source/license record](results/ravdess-mode-matched/license_and_source.md). |
| SUBESCO | [Original publication and data-access information](https://doi.org/10.1371/journal.pone.0250173). The paper's license must not be assumed to license the audio database. Consult the linked dataset's terms when obtaining it. |
| WavLM Base+ | [Microsoft model card](https://huggingface.co/microsoft/wavlm-base-plus) and the version/weight identities in the experiment manifests. |
| HuBERT Base | [Meta/Facebook model card](https://huggingface.co/facebook/hubert-base-ls960) and the version/weight identities in the experiment manifests. |

The Study II recipe ZIP contains selection lists and configuration metadata, not audio. Original providers retain their rights in the underlying databases and recordings. Dataset sources are also cited in the manuscripts; do not cite this repository as the creator of those corpora.

## Conference style and manuscript status

The authors' manuscript source, PDFs and figures are included as research drafts. They are not represented as accepted or published ICASSP papers. Obtain `spconf.sty` from the [official ICASSP 2027 paper kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php) to build the LaTeX source. The third-party style and old ZIPs that bundle it are omitted from this public snapshot; historical build records retain its identity. Other installed TeX packages retain their own licenses.

The public release is named and is not an anonymized submission package. ICASSP 2027's official author instructions state that the conference does not perform blind reviews. Public availability now does not establish that these private pre-execution freezes were public preregistrations.

Historical reports and code may contain a local filesystem location as provenance. Such a path is not a downloadable artifact or a portable input path. Provider account records, credentials, private conversation branches and cloud-control scripts are excluded from the public snapshot.
