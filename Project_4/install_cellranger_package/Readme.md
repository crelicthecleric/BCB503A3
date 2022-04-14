### Install Cellranger package

wget -O cellranger-6.1.2.tar.gz "https://cf.10xgenomics.com/releases/cell-exp/cellranger-6.1.2.tar.gz?Expires=1650000913&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9jZi4xMHhnZW5vbWljcy5jb20vcmVsZWFzZXMvY2VsbC1leHAvY2VsbHJhbmdlci02LjEuMi50YXIuZ3oiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE2NTAwMDA5MTN9fX1dfQ__&Signature=eieGH5RVQTaGJfIidUgX5mbFFK1z9NVCW~5PEnr12ODJ4rGs7YZ843Hir3Nta0n8M627hFu5WFIVvQhdX40u9ZPSGRN4VJNdPXfGSq5TyOZ8Sl4ETDGZS7OP8yfEbLr4vPzztHsx4cYIjVZGF~VnJeUmMi7zQHuTcK7xwaxzxdRNhua86D8YQu0r-tbRW1Z-mILJ9H1KPyD5cuBrWrdmRsj3TL-ZIAqy-O7wqzOHPF5SSXQ6trI-izpNBp7sHo8yaaaQ8vxV61H41haV32VebrO4s9GhiJJY37o~9KaPnzojIm50yitnzlee8leBXAJgyMurCKepnCc~vYzhmyIoOg__&Key-Pair-Id=APKAI7S6A5RYOXBWRPDA" 

tar -xzvf cellranger-6.1.2.tar.gz

### Get human reference genome

wget https://cf.10xgenomics.com/supp/cell-exp/refdata-cellranger-GRCh38-3.0.0.tar.gz
tar -zxvf refdata-cellranger-GRCh38-3.0.0.tar.gz

### Note: Sars_cov.gtf  Sars_cov.fa are the reference genome and annotation for Covid-19

### We then need to combine Human genome and Covid-19 genome together (sample for annotation files, the file with extension .gtf)

The code to combine Human and Covid 19 genome and annotation file: make_ref.sh 


