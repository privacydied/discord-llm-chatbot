"""
URL-based video ingestion module for YouTube/TikTok audio extraction and STT processing.
Integrates with existing hear_infer() pipeline for consistent audio processing.
"""

import os
import re
import asyncio
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

from .utils.logging import get_logger
from .exceptions import InferenceError

logger = get_logger(__name__)

# Configuration from environment
MAX_DURATION_SECONDS = int(os.getenv("VIDEO_MAX_DURATION", "600"))  # 10 minutes default
MAX_CONCURRENT_DOWNLOADS = int(os.getenv("VIDEO_MAX_CONCURRENT", "3"))
CACHE_DIR = Path(os.getenv("VIDEO_CACHE_DIR", "cache/video_audio"))
DEFAULT_SPEEDUP = float(os.getenv("VIDEO_SPEEDUP", "1.5"))
CACHE_EXPIRY_DAYS = int(os.getenv("VIDEO_CACHE_EXPIRY_DAYS", "7"))

# Optional cookies support for yt-dlp to access age/region gated content (e.g., TikTok)
# Provide one of:
#  - VIDEO_COOKIES_FROM_BROWSER="firefox:default-release" (preferred)
#  - VIDEO_COOKIES_FILE="/path/to/cookies.txt" (Netscape format)
# Scope control via VIDEO_COOKIES_SITES (comma-separated, defaults to tiktok only)
YTDLP_COOKIES_FROM_BROWSER = os.getenv("VIDEO_COOKIES_FROM_BROWSER")
YTDLP_COOKIES_FILE = os.getenv("VIDEO_COOKIES_FILE")
YTDLP_COOKIES_SITES = set(
    s.strip().lower()
    for s in os.getenv("VIDEO_COOKIES_SITES", "tiktok").split(",")
    if s.strip()
)

# Supported URL patterns - must match MEDIA_CAPABLE_DOMAINS from media_capability.py
SUPPORTED_PATTERNS = [
    # ---------- YouTube (full set of common forms) ----------
    r"https?://(?:www\.)?youtube\.com/watch\?(?:.*&)?v=[0-9A-Za-z_-]{6,}",
    r"https?://(?:www\.)?youtube\.com/shorts/[0-9A-Za-z_-]{6,}",
    r"https?://(?:www\.)?youtube\.com/(?:live|embed)/[0-9A-Za-z_-]{6,}",
    r"https?://youtu\.be/[0-9A-Za-z_-]{6,}",
    # ---------- TikTok ----------
    r"https?://(?:www\.)?tiktok\.com/@[\w\.-]+/video/\d+",
    r"https?://(?:www\.)?tiktok\.com/t/[\w-]+",  # share links
    r"https?://(?:m|vm)\.tiktok\.com/[\w-]+",
    # ---------- Twitter / X (try yt-dlp first, fallback to screenshot if no video) ----------
    r"https?://(?:www\.)?(?:twitter|x)\.com/\w{1,15}/status/\d+",  # All tweet status URLs - fallback logic will handle non-video tweets
    r"https?://(?:www\.)?(?:twitter|x)\.com/i/broadcasts/\w+",  # Twitter Spaces/Live broadcasts
    # ---------- Reddit (common variants) ----------
    r"https?://(?:www|m)\.reddit\.com/r/[\w-]+/comments/[0-9A-Za-z]+(?:/[\w-]+)?/?",
    r"https?://(?:www|m)\.reddit\.com/video/[0-9A-Za-z_-]+/?",
    r"https?://v\.redd\.it/[0-9A-Za-z]+",
    # ---------- Facebook ----------
    r"https?://(?:www|m|mbasic)\.facebook\.com/(?:[^/?#]+/)?videos/\d+/?",
    r"https?://fb\.watch/[0-9A-Za-z_-]+/?",
    # ---------- Instagram ----------
    r"https?://(?:www\.)?instagram\.com/(?:p|reel|tv)/[0-9A-Za-z_-]+/?",
    r"https?://(?:www\.)?instagram\.com/stories/[^/]+/\d+/?",
    # ---------- Vimeo ----------
    r"https?://(?:www\.)?vimeo\.com/(?:\d+|ondemand/[^/?#]+/[^/?#]+|channels/[^/?#]+/\d+)",
    # ---------- Dailymotion ----------
    r"https?://(?:www\.)?dailymotion\.com/video/[0-9A-Za-z]+",
    # ---------- Twitch ----------
    r"https?://(?:www\.)?twitch\.tv/videos/\d+",
    r"https?://(?:www\.)?twitch\.tv/\w+/clip/[0-9A-Za-z_-]+",
    r"https?://(?:www\.)?twitch\.tv/\w+(?:\?.*)?$",  # live channels
    # ---------- Bilibili ----------
    r"https?://(?:www\.)?bilibili\.com/video/(?:BV[0-9A-Za-z]+|av\d+)",
    r"https?://b23\.tv/[0-9A-Za-z]+",
    # ---------- Rumble / Odysee / LBRY ----------
    r"https?://(?:www\.)?rumble\.com/(?:v|[\w-]+)/[0-9A-Za-z-]+",
    r"https?://(?:www\.)?odysee\.com/@[\w-]+:[\w-]+/[\w-]+:[\w-]+",
    r"https?://(?:www\.)?lbry\.tv/@[\w-]+:[\w-]+/[\w-]+:[\w-]+",
    # ---------- Veoh / Metacafe ----------
    r"https?://(?:www\.)?veoh\.com/watch/[0-9A-Za-z_-]+",
    r"https?://(?:www\.)?metacafe\.com/watch/\d+/[\w-]+",
    # ---------- Sound / Music ----------
    r"https?://(?:www\.)?soundcloud\.com/[\w-]+/[\w-]+",
    r"https?://[\w-]+\.bandcamp\.com/(?:track|album)/[\w-]+",
    r"https?://(?:www\.)?mixcloud\.com/[\w-]+/[\w-]+",
    r"https?://(?:www\.)?audiomack\.com/(?:song|playlist)/[\w-]+/[\w-]+",
    r"https?://open\.spotify\.com/(?:track|album|playlist|episode|show)/[0-9A-Za-z]+",
    # ---------- News / Major broadcasters (commonly requested) ----------
    r"https?://(?:www\.)?cnn\.com/(?:videos?|[^?#]+/video)/[^?#]+",
    r"https?://(?:www\.)?bbc\.co\.uk/(?:iplayer|sounds)/[^?#]+",
    r"https?://(?:www\.)?abc\.net\.au/(?:news|iview)/[^?#]+",
    r"https?://(?:www\.)?nbcnews\.com/video/[^?#]+",
    r"https?://(?:www\.)?foxnews\.com/(?:video|media)/[^?#]+",
    r"https?://(?:www\.)?reuters\.com/video/[^?#]+",
    # ---------- LinkedIn / Pinterest ----------
    r"https?://(?:www\.)?linkedin\.com/(?:posts|feed|learning|video)/[^/?#]+",
    r"https?://(?:www\.)?pinterest\.[a-z.]+/pin/\d+/",
    # ---------- Streamable / VK / Niconico / iQIYI / Viki / VLive ----------
    r"https?://streamable\.com/[0-9A-Za-z]+",
    r"https?://(?:www\.)?vk\.com/(?:video-?\d+_\d+|clip-?\d+_\d+)",
    r"https?://(?:www\.)?nicovideo\.jp/watch/[a-z]{2}\d+",
    r"https?://(?:www\.)?iqiyi\.com/[a-z0-9/_-]+\.html",
    r"https?://(?:www\.)?viki\.com/(?:videos|tv)/[0-9A-Za-z-]+",
    r"https?://(?:www\.)?vlive\.tv/(?:video|post)/\d+",
    # ---------- Adult (explicitly listed in supported sites) ----------
    r"https?://(?:www\.)?pornhub\.com/(?:view_video\.php\?viewkey=|(?:(?:channels|pornstar|model)/[^/]+/)?videos/)\w+",
    r"https?://(?:www\.)?xvideos\.com/video\d+/\w+",
    r"https?://(?:www\.)?xhamster\.com/(?:videos|movies|users/[^/]+/videos)/[0-9A-Za-z-]+",
    # ---------- Massive catch-all union for many additional supported sites ----------
    # Matches ANY path on these domains so you don’t need per-site path rules.
    # Keep this list in sync with yt-dlp/youtube-dl supported sites.
    r"https?://(?:www\.)?(?:"
    r"1tv\.ru|20min\.ch|220\.ro|23video\.com|247sports\.com|24video\.[a-z.]+|3sat\.de|4tube\.com|56\.com|6play\.fr|7plus\.com\.au|"
    r"8tracks\.com|91porn\.com|9gag\.com|9now\.com\.au|abc\.net\.au|abcnews\.go\.com|abc7\.[a-z.]+|acast\.com|adobe(?:tv|connect)\.com|"
    r"afreecatv\.com|aljazeera\.com|allocine\.fr|amara\.org|aparat\.com|apple\.com/trailers|podcasts\.apple\.com|archive\.org|"
    r"ardmediathek\.de|arte\.tv|asiancrush\.com|atresplayer\.com|att\.com|atv\.at|audioboom\.com|awaan\.ae|baidu\.com|bandcamp\.com|"
    r"bangumi\.bilibili\.com|bbc\.co\.uk|bild\.de|bilibili\.com|bitchute\.com|bleacherreport\.com|bloomberg\.com|box\.com|br\.de|"
    r"bravotv\.com|break\.com|brightcove\.(?:com|net)|businessinsider\.com|buzzfeed\.com|byutv\.org|cbc\.ca|cbsnews\.com|cbssports\.com|"
    r"cctv\.com|ceskatelevize\.cz|channel9\.msdn\.com|chaturbate\.com|cielotv\.it|cinemax\.com|cloudflarestream\.com|cmt\.com|cnbc\.com|"
    r"cnn\.com|comedycentral\.(?:com|de|tv)|crackle\.com|crunchyroll\.com|c-span\.org|ctvnews\.ca|curiositystream\.com|cwseed\.com|"
    r"dailymail\.co\.uk|dailymotion\.com|daum\.net|dbtv\.dk|deezer\.com|defense\.gouv\.fr|democracynow\.org|discovery\.(?:com|plus)|"
    r"disney\.(?:com|plus)|dlive\.tv|douyu\.com|dr\.dk|dropbox\.com|dtube\.tv|dumpert\.nl|dw\.com|ebaumsworld\.com|echomsk\.ru|"
    r"egghead\.io|ehftv\.com|ehow\.com|einthusan\.tv|eitb\.eus|ellentube\.com|elpais\.com|embedly\.com|empflix\.com|engadget\.com|"
    r"eporner\.com|eroprofile\.com|escapistmagazine\.com|espn\.com|esri\.com|expressen\.se|extremetube\.com|facebook\.com|fb\.com|"
    r"faz\.net|fc2\.com|filmon\.com|filmweb\.pl|fivethirtyeight\.com|flickr\.com|formula1\.com|fox(?:news|sports)\.com|france\.tv|"
    r"francetvinfo\.fr|freesound\.org|frontendmasters\.com|funimation\.com|gaia\.com|gamespot\.com|giantbomb\.com|gfycat\.com|gogo\.gl|"
    r"globo\.com|godtube\.com|google\.com/drive|hearthis\.at|heise\.de|hgtv\.com|hketv\.hk|hotstar\.com|howcast\.com|huffpost\.com|"
    r"ign\.com|imdb\.com|imgur\.com|ina\.fr|infoq\.com|instagram\.com|internazionale\.it|iprima\.cz|iqiyi\.com|ittf\.com|itv\.com|"
    r"ivi\.ru|ivideon\.com|iwara\.tv|izlesene\.com|jamendo\.com|jeuxvideo\.com|joj\.sk|jwplayer\.com|kakao\.com|kaltura\.com|kankan\.com|"
    r"khanacademy\.org|kickstarter\.com|kinopoisk\.ru|konserthuset\.se|ku6\.com|kusi\.com|kuwo\.cn|la7\.it|laola1\.tv|lbry\.tv|lci\.fr|"
    r"lemonde\.fr|lenta\.ru|libsyn\.com|life\.ru|limelight\.com|line\.me|linetv\.tw|linkedin\.com|linuxacademy\.com|litv\.tv|"
    r"livejournal\.com|livestream\.com|loc\.gov|lrt\.lt|lynda\.com|m6\.fr|mail\.ru|mall\.tv|manyvids\.com|markiza\.sk|matchtv\.ru|"
    r"mdr\.de|medal\.tv|media\.ccc\.de|mediaset\.it|medici\.tv|megaphone\.fm|meipai\.com|metacafe\.com|metacritic\.com|mewatch\.sg|"
    r"mgoon\.com|mgtv\.com|miaopai\.com|minds\.com|ministrygrid\.com|miomio\.tv|mitele\.es|mixcloud\.com|mlb\.com|mnet\.com|"
    r"motherless\.com|motorsport\.com|movieclips\.com|movingimage\.us|msn\.com|mtv\.(?:com|de|co\.uk|jp)|mwave\.me|myspace\.com|"
    r"myspass\.de|myvi\.ru|myvidster\.com|n-tv\.de|nationalgeographic\.com|naver\.com|nba\.com|nbcnews\.com|nbcolympics\.com|"
    r"nbcsports\.com|ndr\.de|ndtv\.com|netflix\.com|netease\.com|netplus\.tv|netzkino\.de|newgrounds\.com|nexttv\.com\.tw|nfl\.com|"
    r"nhk\.or\.jp|nhl\.com|nicovideo\.jp|nintendo\.com|njoy\.de|njpwworld\.com|nobelprize\.org|noovo\.ca|npr\.org|nrk\.no|nrl\.com|"
    r"ntv\.ru|nytimes\.com|nzz\.ch|ocw\.mit\.edu|odnoklassniki\.ru|onet\.pl|ooyala\.com|ora\.tv|orf\.at|outsideonline\.com|packtpub\.com|"
    r"palcomp3\.com\.br|pandora\.tv|paramountnetwork\.com|parliamentlive\.tv|patreon\.com|pbs\.org|peertube\.|people\.com|periscope\.tv|"
    r"philharmoniedeparis\.fr|phoenix\.de|photobucket\.com|picarto\.tv|piksel\.com|pinkbike\.com|pinterest\.[a-z.]+|pladform\.ru|"
    r"platzi\.com|play\.fm|playplus\.com|plays\.tv|play\.idnes\.cz|playvid\.com|playwire\.com|pluralsight\.com|podomatic\.com|"
    r"pokemon\.com|polskieradio\.pl|popcorntimes\.com|popcorntv\.it|pornhub\.com|porntube\.com|redtube\.com|pressTV\.ir|prosiebensat1\.|"
    r"puhutv\.com|qq\.com|qub\.com|quickline\.com|r7\.com|radiocanada\.ca|rai\.it|raiplay\.it|raywenderlich\.com|rbmaradio\.com|"
    r"rds\.ca|redbull\.(?:com|tv)|reddit\.com|regiotv\.de|reuters\.com|reverbnation\.com|rmcdecouverte\.bfmtv\.com|rockstargames\.com|"
    r"rottentomatoes\.com|rtbf\.be|rte\.ie|rtmp|rtve\.es|rtvs\.sk|rutube\.ru|r7\.com|ruhd\.ru|rumble\.com|ruutu\.fi|ruzhe\.|safari(booksonline)?\.|"
    r"sapo\.pt|savefrom\.net|sbs\.com\.au|screencast(?:-o-matic)?\.com|scrippsnetwork|seeker\.com|sendtonews\.com|servus\.com|"
    r"sexu\.com|seznamzpravy\.cz|shahid\.net|shared\.sx|showroom-live\.com|simplecast\.com|sina\.com\.cn|sky\.it|skynewsarabia\.com|"
    r"slideshare\.net|slideslive\.com|slutload\.com|snotr\.com|sohu\.com|sonyliv\.com|soundcloud\.com|soundgasm\.net|southpark\.(?:cc\.com|de|nl)|"
    r"spankbang\.com|spankwire\.com|spiegel\.de|sport\.francetvinfo\.fr|sport5\.co\.il|sportbox\.ru|spotify\.com|spreaker\.com|"
    r"springboardplatform\.com|sproutonline\.com|srf\.ch|stanford\.edu|store\.steampowered\.com|stitcher\.com|storyfire\.com|"
    r"streamable\.com|streamcloud\.eu|streamcz\.cz|streetvoice\.com|stretchinternet\.com|stv\.tv|sunporno\.com|sverigesradio\.se|"
    r"svt(?:play)?\.se|swrmediathek\.de|syfy\.com|tagesschau\.de|tass\.ru|tbs\.com|teachable\.com|teachertube\.com|teachingchannel\.org|"
    r"teamcoco\.com|teamtreehouse\.com|techtalks\.tv|ted\.com|telecinco\.es|teleq(?:uebec|u)\.tv|tenplay\.com\.au|tf1\.fr|tfo\.org|"
    r"theintercept\.com|theplatform\.com|thescene\.com|thesun\.co\.uk|weather\.com|thisamericanlife\.org|thisav\.com|thisoldhouse\.com|"
    r"tiktok\.com|tmz\.com|tnaflix\.com|toggle\.sg|tou\.tv|trailers\.|trilulilu\.ro|trovo\.live|tru(?:(?:news|tv))\.com|tube8\.com|"
    r"tubitv\.com|tumblr\.com|tunein\.com|tunepk\.com|tv(?:2|4|5|8)\.[a-z.]+|tva\.ca|tvc\.ru|tver\.jp|tvigle\.ru|tvland\.com|tvp\.pl|"
    r"tvplayer\.com|tvplay(?:home)?\.|tweakers\.net|twitcasting\.tv|twitch\.tv|udemy\.com|udn\.com|ufc\.|uktvplay\.|"
    r"unity3d\.com|uol\.com\.br|uplynk\.com|urplay\.se|usanetwork\.com|usatoday\.com|ustream\.tv|ustudio\.com|varzesh3\.com|vbox7\.com|"
    r"vee?oh?\.com|vesti\.ru|vevo\.com|vgtv\.no|vh1\.com|viafree\.|vice\.com|viddler\.com|videa\.hu|video\.arnes\.si|video\.sky\.it|"
    r"videodetective\.com|videomore\.ru|videopress\.com|vidio\.com|vidlii\.com|vier\.be|viewlift\.com|viidea\.fi|viki\.com|vimeo\.com|"
    r"vimple\.ru|vine\.co|viqeo\.tv|viu\.(?:com|tv)|vivo\.sx|vk\.com|vlive\.tv|vodlocker\.com|voice\.republic\.|voot\.com|voxmedia\.|"
    r"vrt\.be|vrv\.co|vshare\.io|vtm\.be|vtx\.ch|vuclip\.com|vvvvid\.it|vzaar\.com|wakanim\.tv|walla\.co\.il|washingtonpost\.com|"
    r"wat\.tv|watchbox\.de|watchindianporn\.|wdr\.de|webcaster\.|webofstories\.com|weibo\.com|wistia\.(?:com|net)|worldstarhiphop\.com|"
    r"wsj\.com|wwe\.com|xbef\.|xboxclips\.com|xfileshare\.|xhamster\.com|xiami\.com|ximalaya\.com|xminus\.me|xnxx\.com|xstream\.|"
    r"xtube\.com|xuite\.net|xvideos\.com|xxxy\.|yahoo\.(?:com|co\.jp)|yandex\.(?:ru|com)|yandex\.music|yandex\.video|yapfiles\.ru|"
    r"yesjapan\.com|yinyuetai\.com|ynet\.co\.il|youjizz\.com|youku\.com|younow\.com|youporn\.com|yourporn\.se|yourupload\.com|"
    r"youtube\.com|youtu\.be|zapiks\.fr|zattoo\.com|zdf\.de|zhihu\.com|zingmp3\.vn|zoom\.us|zype\.com"
    r")/[^\s>]+",
]


# Global semaphore for download concurrency
_download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)


@dataclass
class VideoMetadata:
    """Metadata extracted from video source."""

    url: str
    title: str
    duration_seconds: float
    uploader: str
    upload_date: str
    source_type: str  # 'youtube' or 'tiktok'


@dataclass
class ProcessedAudio:
    """Result of video audio processing."""

    audio_path: Path
    metadata: VideoMetadata
    processed_duration_seconds: float
    speedup_factor: float
    cache_hit: bool
    timestamp: datetime


class VideoIngestError(InferenceError):
    """Specific error for video ingestion failures."""

    pass


class VideoIngestionManager:
    """Manages video URL ingestion, caching, and audio processing."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._setup_cache_index()
        logger.info(
            f"🎥 VideoIngestionManager initialized with cache: {self.cache_dir}"
        )

    def _setup_cache_index(self):
        """Initialize cache index file."""
        self.cache_index_path = self.cache_dir / "index.json"
        if not self.cache_index_path.exists():
            with open(self.cache_index_path, "w") as f:
                json.dump({}, f)

    def _get_cache_key(self, url: str) -> str:
        """Generate deterministic cache key for URL."""
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    def _is_supported_url(self, url: str) -> bool:
        """Check if URL matches supported patterns."""
        return any(re.match(pattern, url) for pattern in SUPPORTED_PATTERNS)

    def _get_source_type(self, url: str) -> str:
        """Determine source type from URL."""
        if "youtube.com" in url or "youtu.be" in url:
            return "youtube"
        elif "tiktok.com" in url:
            return "tiktok"
        else:
            return "unknown"

    async def _download_with_ytdlp(
        self, url: str, output_path: Path
    ) -> Tuple[VideoMetadata, Path]:
        """Download video audio using yt-dlp."""
        logger.info(f"📥 Downloading audio from: {url}")

        def _should_apply_cookies(u: str) -> bool:
            source = self._get_source_type(u)
            return (
                bool(YTDLP_COOKIES_FROM_BROWSER or YTDLP_COOKIES_FILE)
                and (not YTDLP_COOKIES_SITES or source in YTDLP_COOKIES_SITES)
            )

        def _maybe_with_cookies(base_cmd: list, u: str) -> list:
            """Append cookies args if configured and within scope.

            Returns a new list (does not mutate the input).
            """
            cmd = list(base_cmd)
            if _should_apply_cookies(u):
                if YTDLP_COOKIES_FROM_BROWSER:
                    cmd += ["--cookies-from-browser", YTDLP_COOKIES_FROM_BROWSER]
                    logger.debug(
                        "🔑 Applying cookies from browser for yt-dlp (site scope: %s)",
                        ",".join(sorted(YTDLP_COOKIES_SITES)) or "<all>",
                    )
                elif YTDLP_COOKIES_FILE:
                    cmd += ["--cookies", YTDLP_COOKIES_FILE]
                    logger.debug(
                        "🔑 Applying cookies from file for yt-dlp (site scope: %s)",
                        ",".join(sorted(YTDLP_COOKIES_SITES)) or "<all>",
                    )
            else:
                logger.debug(
                    "ℹ️ Not applying cookies (configured: %s, site: %s, scope: %s)",
                    bool(YTDLP_COOKIES_FROM_BROWSER or YTDLP_COOKIES_FILE),
                    self._get_source_type(u),
                    ",".join(sorted(YTDLP_COOKIES_SITES)) or "<all>",
                )
            return cmd

        # First get metadata with JSON output for reliable parsing
        metadata_cmd = ["yt-dlp", "--dump-json", "--no-playlist", "--quiet", url]
        metadata_cmd = _maybe_with_cookies(metadata_cmd, url)
        logger.debug("🧰 yt-dlp metadata cmd: %s", " ".join(metadata_cmd[:-1] + ["<URL>"]))

        try:
            # Get metadata first
            proc = await asyncio.create_subprocess_exec(
                *metadata_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            # Bound metadata probe to avoid long hangs [REH][PA]
            try:
                md_timeout = float(os.getenv("YTDLP_METADATA_TIMEOUT_S", "12"))
            except Exception:
                md_timeout = 12.0
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=md_timeout
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass
                raise VideoIngestError(
                    f"yt-dlp metadata probe timed out after {md_timeout:.0f}s"
                )

            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown yt-dlp error"

                # Handle specific cases where no video/audio content is found
                if any(
                    phrase in error_msg.lower()
                    for phrase in [
                        "no video could be found",
                        "no video formats found",
                        "no audio could be found",
                        "no extractors found",
                        "unable to extract video info",
                        "private video",
                        "video not available",
                        "no video",
                        "no audio",
                    ]
                ):
                    logger.info(f"ℹ️ No video/audio content found in URL: {url}")
                    raise VideoIngestError(
                        "No video or audio content found in this URL. This might be a text-only post or unavailable content."
                    )

                # Provide targeted guidance for authentication-gated content
                if any(
                    k in error_msg.lower()
                    for k in [
                        "log in for access",
                        "use --cookies-from-browser",
                        "use --cookies for the authentication",
                        "age-restricted",
                    ]
                ):
                    logger.warning(
                        "⚠️ yt-dlp requires authentication for this URL. Configure VIDEO_COOKIES_FROM_BROWSER or VIDEO_COOKIES_FILE env vars."
                    )
                raise VideoIngestError(
                    f"yt-dlp metadata extraction failed: {error_msg}"
                )

            # Parse JSON metadata
            metadata_json = json.loads(stdout.decode())

            # Extract metadata with safe defaults
            title = metadata_json.get("title", "Unknown Title")
            duration = float(metadata_json.get("duration", 0.0) or 0.0)
            uploader = metadata_json.get("uploader", "Unknown")
            upload_date = metadata_json.get("upload_date", "")

            # Now download the audio
            download_cmd = [
                "yt-dlp",
                "--extract-audio",
                "--audio-format",
                "wav",
                "--audio-quality",
                "0",  # Best quality
                "--no-playlist",
                "--output",
                str(output_path / "%(title)s.%(ext)s"),
                "--print",
                "after_move:filepath",
                "--quiet",
                url,
            ]
            download_cmd = _maybe_with_cookies(download_cmd, url)
            logger.debug(
                "🧰 yt-dlp download cmd: %s", " ".join(download_cmd[:-1] + ["<URL>"])
            )

            proc = await asyncio.create_subprocess_exec(
                *download_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            # Bound download to avoid indefinite hangs [REH][PA]
            try:
                dl_timeout = float(os.getenv("YTDLP_DOWNLOAD_TIMEOUT_S", "90"))
            except Exception:
                dl_timeout = 90.0
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=dl_timeout
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass
                raise VideoIngestError(
                    f"yt-dlp download timed out after {dl_timeout:.0f}s"
                )

            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown yt-dlp error"
                if any(
                    k in error_msg.lower()
                    for k in [
                        "log in for access",
                        "use --cookies-from-browser",
                        "use --cookies for the authentication",
                        "age-restricted",
                    ]
                ):
                    logger.warning(
                        "⚠️ yt-dlp requires authentication for this URL. Configure VIDEO_COOKIES_FROM_BROWSER or VIDEO_COOKIES_FILE env vars."
                    )
                raise VideoIngestError(f"yt-dlp download failed: {error_msg}")

            # Get the filepath from output
            filepath = stdout.decode().strip()
            if not filepath:
                raise VideoIngestError("No filepath returned from yt-dlp")

            return VideoMetadata(
                url=url,
                title=title,
                duration_seconds=duration,
                uploader=uploader,
                upload_date=upload_date,
                source_type=self._get_source_type(url),
            ), Path(filepath)

        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse yt-dlp JSON metadata: {e}")
            raise VideoIngestError(f"Failed to parse video metadata: {str(e)}")
        except Exception as e:
            # Log at INFO level for expected "no video content" errors to avoid scary tracebacks
            error_str = str(e).lower()
            if any(
                expected_error in error_str
                for expected_error in [
                    "no video could be found",
                    "no video formats found",
                    "no audio could be found",
                    "no extractors found",
                    "unable to extract video info",
                    "private video",
                    "video not available",
                    "no video",
                    "no audio",
                ]
            ):
                logger.info(f"ℹ️ yt-dlp: {e}")
            else:
                # Only log unexpected errors with tracebacks
                logger.error(f"❌ yt-dlp download failed: {e}", exc_info=True)
            raise VideoIngestError(f"Failed to download video: {str(e)}")

    async def _process_audio(
        self, raw_audio_path: Path, speedup: float = DEFAULT_SPEEDUP
    ) -> Path:
        """Process raw audio with same normalization as hear_infer()."""
        logger.info(f"🔄 Processing audio with {speedup}x speedup")

        # Create processed audio path
        processed_path = raw_audio_path.with_suffix(".processed.wav")

        # FFmpeg command matching hear_infer() logic
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_audio_path),
            "-ar",
            "16000",  # 16kHz sample rate
            "-ac",
            "1",  # Mono
            "-acodec",
            "pcm_s16le",  # 16-bit PCM
            "-af",
            f"atempo={speedup},aresample=async=1:first_pts=0",  # Speed + resample
            str(processed_path),
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            # Bound ffmpeg processing to avoid hangs [REH][PA]
            try:
                ff_timeout = float(os.getenv("FFMPEG_AUDIO_PROC_TIMEOUT_S", "60"))
            except Exception:
                ff_timeout = 60.0
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=ff_timeout
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass
                raise VideoIngestError(
                    f"Audio processing timed out after {ff_timeout:.0f}s"
                )

            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown ffmpeg error"
                raise VideoIngestError(f"Audio processing failed: {error_msg}")

            logger.info(f"✅ Audio processed: {processed_path}")
            return processed_path

        except Exception as e:
            logger.error(f"❌ Audio processing failed: {e}")
            raise VideoIngestError(f"Failed to process audio: {str(e)}")

    def _update_cache_index(
        self, cache_key: str, metadata: VideoMetadata, processed_path: Path
    ):
        """Update cache index with new entry."""
        try:
            with open(self.cache_index_path, "r") as f:
                index = json.load(f)

            index[cache_key] = {
                "url": metadata.url,
                "title": metadata.title,
                "duration_seconds": metadata.duration_seconds,
                "uploader": metadata.uploader,
                "upload_date": metadata.upload_date,
                "source_type": metadata.source_type,
                "processed_path": str(processed_path),
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "speedup_factor": DEFAULT_SPEEDUP,
            }

            with open(self.cache_index_path, "w") as f:
                json.dump(index, f, indent=2)

        except Exception as e:
            logger.warning(f"⚠️ Failed to update cache index: {e}")

    def _get_cached_entry(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached entry if it exists and is valid."""
        try:
            with open(self.cache_index_path, "r") as f:
                index = json.load(f)

            if cache_key not in index:
                return None

            entry = index[cache_key]
            processed_path = Path(entry["processed_path"])

            # Check if cached file exists
            if not processed_path.exists():
                logger.warning(f"⚠️ Cached file missing: {processed_path}")
                return None

            # Check cache expiry
            cached_at = datetime.fromisoformat(entry["cached_at"])
            age_days = (datetime.now(timezone.utc) - cached_at).days

            if age_days > CACHE_EXPIRY_DAYS:
                logger.info(f"🗑️ Cache entry expired ({age_days} days old)")
                return None

            return entry

        except Exception as e:
            logger.warning(f"⚠️ Failed to read cache index: {e}")
            return None

    async def fetch_and_prepare_url_audio(
        self, url: str, speedup: float = DEFAULT_SPEEDUP, force_refresh: bool = False
    ) -> ProcessedAudio:
        """
        Main entry point: fetch video URL and prepare audio for STT pipeline.

        Args:
            url: YouTube or TikTok URL
            speedup: Audio speedup factor (default 1.5x)
            force_refresh: Force re-download even if cached

        Returns:
            ProcessedAudio object ready for hear_infer()
        """
        if not self._is_supported_url(url):
            raise VideoIngestError(f"Unsupported URL format: {url}")

        cache_key = self._get_cache_key(url)

        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_entry = self._get_cached_entry(cache_key)
            if cached_entry:
                logger.info(f"💾 Cache hit for: {url}")

                metadata = VideoMetadata(
                    url=cached_entry["url"],
                    title=cached_entry["title"],
                    duration_seconds=cached_entry["duration_seconds"],
                    uploader=cached_entry["uploader"],
                    upload_date=cached_entry["upload_date"],
                    source_type=cached_entry["source_type"],
                )

                return ProcessedAudio(
                    audio_path=Path(cached_entry["processed_path"]),
                    metadata=metadata,
                    processed_duration_seconds=cached_entry["duration_seconds"]
                    / speedup,
                    speedup_factor=speedup,
                    cache_hit=True,
                    timestamp=datetime.now(timezone.utc),
                )

        # Download and process new content
        async with _download_semaphore:
            logger.info(f"🔄 Processing new URL: {url}")

            # Create temporary directory for this download
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                try:
                    # Download with yt-dlp
                    metadata, raw_audio_path = await self._download_with_ytdlp(
                        url, temp_path
                    )

                    # Validate duration
                    if metadata.duration_seconds > MAX_DURATION_SECONDS:
                        raise VideoIngestError(
                            f"Video too long: {metadata.duration_seconds:.1f}s "
                            f"(max: {MAX_DURATION_SECONDS}s)"
                        )

                    # Process audio (normalize + speedup)
                    processed_path = await self._process_audio(raw_audio_path, speedup)

                    # Move to cache directory (handle cross-filesystem moves)
                    cache_audio_path = self.cache_dir / f"{cache_key}.wav"
                    shutil.move(str(processed_path), str(cache_audio_path))

                    # Update cache index
                    self._update_cache_index(cache_key, metadata, cache_audio_path)

                    logger.info(f"✅ Successfully processed: {metadata.title}")

                    return ProcessedAudio(
                        audio_path=cache_audio_path,
                        metadata=metadata,
                        processed_duration_seconds=metadata.duration_seconds / speedup,
                        speedup_factor=speedup,
                        cache_hit=False,
                        timestamp=datetime.now(timezone.utc),
                    )

                except Exception as e:
                    # Check if this is an expected VideoIngestError or unexpected error
                    if isinstance(e, VideoIngestError):
                        # Don't log VideoIngestError again - it was already logged appropriately in _download_with_ytdlp
                        logger.debug(f"VideoIngestError propagating: {e}")
                    else:
                        logger.error(
                            f"❌ Failed to process URL {url}: {e}", exc_info=True
                        )
                    raise


# Global instance
video_manager = VideoIngestionManager()


async def fetch_and_prepare_url_audio(
    url: str, speedup: float = DEFAULT_SPEEDUP, force_refresh: bool = False
) -> ProcessedAudio:
    """
    Convenience function to fetch and prepare URL audio.

    Args:
        url: YouTube or TikTok URL
        speedup: Audio speedup factor (default 1.5x)
        force_refresh: Force re-download even if cached

    Returns:
        ProcessedAudio object ready for STT pipeline
    """
    return await video_manager.fetch_and_prepare_url_audio(url, speedup, force_refresh)
