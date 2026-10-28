"""GTFtoSeq extracts the same CDS, protein and flanks for an ORF whatever order its CDS rows are listed in,
on both strands. Expected sequences are built directly from the genome by position.

Run with pytest, or directly: python tests/test_gtf_to_seq.py
"""
import random

import pandas as pd
from Bio.Seq import Seq

import shortstop.pipeline  # noqa: F401  (import order used by the package; importing converters first is circular)
from shortstop.converters.gtf_to_seq import GTFtoSeq

UTR = 25
CODONS = ['GCT', 'AAA', 'CTG', 'GAA', 'TCC', 'CGT', 'ACC', 'GGC', 'ATC']  # no stop codons


def revcomp(s):
    return str(Seq(s).reverse_complement())


def build_genome():
    """300 nt chrT with a plus-strand ORF p1 (CDS 41-55 + 71-85, stop 86-88) and a minus-strand ORF m1
    (CDS 191-205 + 221-235 read 235 -> 191, stop 188-190), each 10 codons, in transcripts 11-140 and 151-280."""
    rng = random.Random(0)
    g = [rng.choice('ACGT') for _ in range(300)]

    def put(start, seq):  # 1-based
        g[start - 1:start - 1 + len(seq)] = list(seq)

    coding = 'ATG' + ''.join(rng.choice(CODONS) for _ in range(9))  # 30 nt
    put(41, coding[:15]); put(71, coding[15:]); put(86, 'TAA')
    m_coding = 'ATG' + ''.join(rng.choice(CODONS) for _ in range(9))
    m_genomic = revcomp(m_coding)  # the CDS read low to high on the genome: 191..205 then 221..235
    put(191, m_genomic[:15]); put(221, m_genomic[15:]); put(188, revcomp('TAA'))
    return ''.join(g), coding, m_coding


def gtf_rows(minus_descending, plus_descending):
    attr = lambda i: f'gene_id "{i}"; transcript_id "{i}";'
    rows = [['chrT', 't', 'transcript', 11, 140, '.', '+', '.', attr('p1')],
            ['chrT', 't', 'transcript', 151, 280, '.', '-', '.', attr('m1')]]
    p = [['chrT', 't', 'CDS', 41, 55, '.', '+', '0', attr('p1')], ['chrT', 't', 'CDS', 71, 85, '.', '+', '0', attr('p1')]]
    m = [['chrT', 't', 'CDS', 191, 205, '.', '-', '0', attr('m1')], ['chrT', 't', 'CDS', 221, 235, '.', '-', '0', attr('m1')]]
    return rows + (p[::-1] if plus_descending else p) + (m[::-1] if minus_descending else m)


def expected(genome, coding, m_coding):
    g = lambda s, e: genome[s - 1:e]
    return {
        '"p1"': dict(cds_seq=coding, aa_seq=str(Seq(coding).translate()), cds_starts=41, cds_ends=85,
                     utr_5=g(41 - UTR, 40), utr_3=g(89, 89 + UTR - 1)),
        '"m1"': dict(cds_seq=m_coding, aa_seq=str(Seq(m_coding).translate()), cds_starts=235, cds_ends=191,
                     utr_5=revcomp(g(236, 236 + UTR - 1)), utr_3=revcomp(g(187 - UTR + 1, 187))),
    }


def run(tmp_path, minus_descending, plus_descending):
    genome, coding, m_coding = build_genome()
    fa = tmp_path / 'g.fa'
    fa.write_text('>chrT\n' + genome + '\n')
    gtf = tmp_path / f'orfs_{minus_descending}_{plus_descending}.gtf'
    pd.DataFrame(gtf_rows(minus_descending, plus_descending)).to_csv(gtf, sep='\t', header=False, index=False)
    out = GTFtoSeq(gtf_file=str(gtf), fasta_file=str(fa), utr_length=UTR).extract_sequences().set_index('orf_id')
    return out, expected(genome, coding, m_coding)


def check(out, exp):
    assert set(out.index) == set(exp)
    for orf, cols in exp.items():
        for col, value in cols.items():
            assert out.at[orf, col] == value, (orf, col, out.at[orf, col], value)


def test_ascending_rows(tmp_path):
    check(*run(tmp_path, minus_descending=False, plus_descending=False))


def test_transcript_order_rows(tmp_path):  # GENCODE style: minus-strand CDS rows listed high to low
    check(*run(tmp_path, minus_descending=True, plus_descending=False))


def test_row_order_does_not_matter(tmp_path):
    outs = [run(tmp_path, m, p)[0] for m in (False, True) for p in (False, True)]
    for o in outs[1:]:
        pd.testing.assert_frame_equal(o, outs[0])


if __name__ == '__main__':
    import pathlib, tempfile
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            with tempfile.TemporaryDirectory() as d:
                fn(pathlib.Path(d))
            print('ok', name)
