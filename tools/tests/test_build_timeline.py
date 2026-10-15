import json, subprocess, sys, tempfile, unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures"


def build(out, *extra):
    cmd = [sys.executable, str(TOOLS / "build_timeline.py"), "--allin1", str(FIX / "song.json"),
           "--energy", str(FIX / "librosa.json"), "--srt", str(FIX / "lyrics.srt"),
           "--audio", str(FIX / "fake_audio.txt"), "-o", str(out), *extra]
    subprocess.run(cmd, check=True, capture_output=True)
    return json.loads(Path(out).read_text(encoding="utf-8"))


class BuildTimeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "timeline.json"

    def tearDown(self):
        self.tmp.cleanup()

    def check(self):
        r = subprocess.run([sys.executable, str(TOOLS / "check_timeline.py"), str(self.out), "--fixture"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_basic(self):
        t = build(self.out)
        self.assertEqual(t["meta"]["n_frames"], 240)
        self.assertEqual(t["meta"]["bpm_source"], "allin1")
        self.assertTrue(any("2x" in w for w in t["meta"]["warnings"]))
        self.assertEqual(t["meta"]["irregular_bars"], [{"bar": 5, "beats": 2, "reason": "song end"}])
        self.assertFalse(any("irregular" in w for w in t["meta"]["warnings"]))
        self.assertEqual(t["hits"], [])
        self.assertEqual([b["bar"] for b in t["beats"][:3]], [0, 0, 1])
        self.assertEqual(t["bars"][0]["frame_start"], 24)
        self.assertEqual(t["bars"][-1]["frame_end"], 240)
        self.assertEqual(max(b["energy"] for b in t["bars"]), 10.0)
        self.assertEqual([s["label"] for s in t["sections"]], ["intro", "chorus1", "verse", "chorus2"])
        s = t["sections"]
        self.assertEqual(s[0]["frame_start"], 0)
        self.assertEqual(s[-1]["frame_end"], 240)
        self.assertEqual(s[1]["t_start"], 3.0)
        self.assertEqual(s[1]["t_start_raw"], 2.9)
        self.assertEqual(s[1]["snap_beats"], 0.2)
        self.assertEqual(t["lyrics"][1]["text"], "你好 world")
        self.assertEqual(t["lyrics"][0]["bar_start"], 0)
        self.assertEqual(t["lyrics"][1]["bar_start"], 2)
        self.assertIn(2, [b["bar"] for b in t["bars"] if 2 in b["lyric_ids"]])
        self.check()

    def test_double_halve(self):
        t = build(self.out, "--bpm-mode", "double")
        self.assertEqual(t["meta"]["bpm"], 240)
        self.assertEqual(t["meta"]["bpm_source"], "allin1_doubled")
        self.assertEqual(len(t["beats"]), 39)
        self.assertAlmostEqual(t["bars"][0]["t_end"] - t["bars"][0]["t_start"], 1.0)
        self.check()
        t = build(self.out, "--bpm-mode", "halve")
        self.assertEqual(t["meta"]["bpm_source"], "allin1_halved")
        self.assertAlmostEqual(t["bars"][0]["t_end"] - t["bars"][0]["t_start"], 4.0)
        self.check()

    def test_override(self):
        t = build(self.out, "--sections", str(FIX / "sections.json"))
        self.assertEqual(t["meta"]["sections_source"], "override:" + str(FIX / "sections.json"))
        self.assertIsNotNone(t["meta"]["sources"]["sections"]["sha256"])
        self.assertEqual([s["label"] for s in t["sections"]], ["a", "b", "c"])
        self.assertEqual([s["t_start"] for s in t["sections"]], [0.0, 3.0, 7.0])
        self.assertEqual(t["sections"][1]["alt_bar"], 3)
        self.assertEqual(t["sections"][1]["alt_t"], 5.0)
        self.assertTrue(all(s["snap_beats"] == 0 for s in t["sections"]))
        self.assertEqual([s["label"] for s in t["sections_allin1"]], ["intro", "chorus1", "verse", "chorus2"])
        self.assertEqual(t["bars"][1]["section"], "b")  # bar 2 starts at 3.0
        self.assertEqual(t["lyrics"][2]["section"], "c")
        self.check()

    def test_override_invalid(self):
        bad = Path(self.tmp.name) / "bad.json"
        bad.write_text(json.dumps({"sections": [{"label": "x", "bar": 1}]}))
        with self.assertRaises(subprocess.CalledProcessError):
            build(self.out, "--sections", str(bad))

    def test_beat_corrections(self):
        cf = Path(self.tmp.name) / "corr.json"
        cf.write_text(json.dumps({"corrections": [
            {"region": "r3", "allin1_t": [3.0, 3.5, 4.0, 4.5], "new_beats_t": [3.0, 3.6, 4.2],
             "new_is_downbeat": [True, False, False]},
            {"region": "r5", "allin1_t": [5.5], "new_beats_t": [5.3, 5.7], "new_is_downbeat": [False, False]}]}))
        t = build(self.out, "--beat-corrections", str(cf))
        self.assertEqual(t["meta"]["beat_corrections"]["regions"], ["r3", "r5"])
        self.assertIn("beat_corrections", t["meta"]["sources"])
        bars = {}
        for b in t["beats"]:
            bars.setdefault(b["bar"], []).append(b)
        self.assertEqual([b["beat_in_bar"] for b in bars[2]], [1, 2, 3])
        self.assertEqual([b["beat_in_bar"] for b in bars[3]], [1, 2, 3, 4, 5])
        self.assertEqual([b["t"] for b in bars[2]], [3.0, 3.6, 4.2])
        self.assertEqual(sorted(b["t"] for b in t["beats"] if b.get("corrected")), [3.0, 3.6, 4.2, 5.3, 5.7])
        self.assertEqual({x["bar"]: (x["beats"], x["reason"]) for x in t["meta"]["irregular_bars"]},
                         {2: (3, "beat_corrections:r3"), 3: (5, "beat_corrections:r5"), 5: (2, "song end")})
        self.assertFalse(any("irregular" in w for w in t["meta"]["warnings"]))
        self.check()

    def test_beat_corrections_unmatched(self):
        cf = Path(self.tmp.name) / "corr.json"
        cf.write_text(json.dumps({"corrections": [
            {"region": "x", "allin1_t": [3.25], "new_beats_t": [3.3], "new_is_downbeat": [False]}]}))
        with self.assertRaises(subprocess.CalledProcessError):
            build(self.out, "--beat-corrections", str(cf))

    def test_hits(self):
        sf = Path(self.tmp.name) / "sec.json"
        sf.write_text(json.dumps({"sections": [{"label": "a", "bar": 0}, {"label": "b", "bar": 2}],
                                  "hits": [{"bar": 3, "type": "drop", "basis": "test"}]}))
        t = build(self.out, "--sections", str(sf))
        self.assertEqual(t["hits"], [{"bar": 3, "type": "drop", "basis": "test", "t": 5.0, "frame": 120, "section": "b"}])
        self.assertEqual(t["bars"][2]["hit"], "drop")
        self.assertNotIn("hit", t["bars"][1])
        self.check()
        sf.write_text(json.dumps({"sections": [{"label": "a", "bar": 0}], "hits": [{"bar": 9, "type": "drop"}]}))
        with self.assertRaises(subprocess.CalledProcessError):
            build(self.out, "--sections", str(sf))


if __name__ == "__main__":
    unittest.main()
