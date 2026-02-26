jixabot@Jixabotdiscord:~/blokdowloder$ cat src/downloader.py | grep -n "before_files"
118:        before_files = set(self.temp_dir.glob("*.mp3"))
152:        new_files = after_files - before_files
314:    def _collect_new_files(self, before_files: set[Path]) -> list[Path]:
315:        """Return list of new audio files created since before_files snapshot."""
318:        new_files = sorted(after_files - before_files, key=lambda f: f.stat().st_mtime)
357:        before_files = set(self.temp_dir.glob(pattern))
363:        new_files = self._collect_new_files(before_files)
408:        before_files = set(self.temp_dir.glob(pattern))
414:        new_files = self._collect_new_files(before_files)
jixabot@Jixabotdiscord:~/blokdowloder$
