"""URL-based video ingestion module for YouTube/TikTok audio extraction and STT processing.
Integrates with existing hear_infer() pipeline for consistent audio processing.
"""

import asyncio
import contextlib
import hashlib
import html
import json
import os
import re
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import ParseResult, urlparse, urlunparse

from .config import load_config
from .exceptions import InferenceError
from .utils.external_api import _is_private_hostname
from .utils.logging import get_logger

logger = get_logger(__name__)

# Configuration from environment
MAX_DURATION_SECONDS = int(os.getenv("VIDEO_MAX_DURATION", "600"))  # 10 minutes default
MAX_CONCURRENT_DOWNLOADS = int(os.getenv("VIDEO_MAX_CONCURRENT", "3"))
CACHE_DIR = Path(os.getenv("VIDEO_CACHE_DIR", "cache/video_audio"))
CACHE_EXPIRY_DAYS = int(os.getenv("VIDEO_CACHE_EXPIRY_DAYS", "7"))

# Optional cookies support for yt-dlp to access age/region gated content (e.g., TikTok)
# Provide one of:
#  - VIDEO_COOKIES_FROM_BROWSER="firefox:default-release" (preferred)
#  - VIDEO_COOKIES_FILE="/path/to/cookies.txt" (Netscape format)
# Scope control via VIDEO_COOKIES_SITES (comma-separated, defaults to tiktok only)
YTDLP_COOKIES_FROM_BROWSER = os.getenv("VIDEO_COOKIES_FROM_BROWSER")
YTDLP_COOKIES_FILE = os.getenv("VIDEO_COOKIES_FILE")
YTDLP_COOKIES_SITES = {s.strip().lower() for s in os.getenv("VIDEO_COOKIES_SITES", "tiktok").split(",") if s.strip()}

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
    r"https?://(?:www\.)?(?:twitter|x|fxtwitter|vxtwitter|fixupx)\.com/\w{1,15}/status/\d+",  # All tweet status URLs - fallback logic will handle non-video tweets
    r"https?://(?:www\.)?(?:twitter|x|fxtwitter|vxtwitter|fixupx)\.com/i/broadcasts/\w+",  # Twitter Spaces/Live broadcasts
    # ---------- Twitter CDN direct variants (mp4/HLS from vx/fx/native) ----------
    r"https?://(?:video|mtc)\.twimg\.com/[^\s?#]+\.(?:mp4|m3u8)(?:\?[^\s#]+)?",
    # ---------- Reddit (common variants) ----------
    r"https?://(?:www|m)\.reddit\.com/r/[\w-]+/comments/[0-9A-Za-z]+(?:/[\w-]+)?/?",
    r"https?://(?:www|m)\.reddit\.com/video/[0-9A-Za-z_-]+/?",
    r"https?://v\.redd\.it/[0-9A-Za-z]+",
    # ---------- Facebook ----------
    r"https?://(?:www|m|mbasic)\.facebook\.com/(?:[^/?#]+/)?videos/\d+/?",
    r"https?://fb\.watch/[0-9A-Za-z_-]+/?",
    # ---------- Instagram ----------
    r"https?://(?:www\.)?(?:instagram|kkinstagram)\.com/(?:p|reel|tv)/[0-9A-Za-z_-]+/?",
    r"https?://d\.vxinstagram\.com/(?:p|reel|tv)/[0-9A-Za-z_-]+/?",
    r"https?://d\.vxinstagram\.com/offload/[^\s?#]+\.(?:mp4|m4a|aac|opus|ogg|mp3|webm)(?:\?[^\s#]+)?",
    r"https?://(?:www\.)?(?:instagram|kkinstagram)\.com/stories/[^/]+/\d+/?",
    r"https?://d\.vxinstagram\.com/stories/[^/]+/\d+/?",
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

# Domain-to-extractor mapping for metadata validation and identity canonicalization [CMV]
_DOMAIN_EXTRACTOR_MAP: dict[str, str] = {
    "youtube.com": "youtube",
    "youtu.be": "youtube",
    "www.youtube.com": "youtube",
    "m.youtube.com": "youtube",
    "tiktok.com": "tiktok",
    "www.tiktok.com": "tiktok",
    "vm.tiktok.com": "tiktok",
    "m.tiktok.com": "tiktok",
    "twitter.com": "twitter",
    "www.twitter.com": "twitter",
    "x.com": "twitter",
    "www.x.com": "twitter",
    "fxtwitter.com": "twitter",
    "vxtwitter.com": "twitter",
    "fixupx.com": "twitter",
    "instagram.com": "instagram",
    "www.instagram.com": "instagram",
    "kkinstagram.com": "instagram",
    "www.kkinstagram.com": "instagram",
    "d.vxinstagram.com": "instagram",
    "reddit.com": "reddit",
    "www.reddit.com": "reddit",
    "v.redd.it": "reddit",
    "vimeo.com": "vimeo",
    "www.vimeo.com": "vimeo",
    "twitch.tv": "twitch",
    "www.twitch.tv": "twitch",
    "dailymotion.com": "dailymotion",
    "www.dailymotion.com": "dailymotion",
    "facebook.com": "facebook",
    "www.facebook.com": "facebook",
    "fb.watch": "facebook",
    "soundcloud.com": "soundcloud",
    "www.soundcloud.com": "soundcloud",
    "bilibili.com": "bilibili",
    "www.bilibili.com": "bilibili",
    "b23.tv": "bilibili",
}


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
class DownloadedAudio:
    """Raw audio artifact fetched via yt-dlp prior to preprocessing."""

    raw_path: Path
    metadata: VideoMetadata
    download_key: str
    format_id: str
    resolved_url: str
    content_length: int | None
    cache_hit: bool
    ext: str
    timestamp: datetime
    demux_fallback: bool = False


@dataclass
class ProcessedAudio:
    """Backward-compatible audio artifact shape for legacy tests/callers. [CA][REH].

    NOTE: The current STT pipeline consumes DownloadedAudio + preprocess stage.
    This shim exists to keep older imports from breaking.
    """

    audio_path: Path
    metadata: VideoMetadata
    processed_duration_seconds: float = 0.0
    speedup_factor: float = 1.0
    cache_hit: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    demux_fallback: bool = False

    @property
    def raw_path(self) -> Path:
        return self.audio_path


class VideoIngestError(InferenceError):
    """Specific error for video ingestion failures."""



class VideoIngestionManager:
    """Manages video URL ingestion, caching, and raw audio acquisition."""

    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index_path = self.cache_dir / "index.json"
        self._index: dict[str, dict[str, Any]] = self._load_cache_index()
        # Ensure cache index exists for deterministic behavior in tests/runtime. [RM]
        if not self.cache_index_path.exists():
            self._save_cache_index()
        cfg = load_config()
        try:
            max_mb = int(cfg.get("MAX_ATTACHMENT_SIZE_MB", 25))
        except Exception:
            max_mb = 25
        self._size_guard_bytes = max_mb * 1024 * 1024
        logger.info(f"🎥 VideoIngestionManager initialized with cache={self.cache_dir} size_guard={self._size_guard_bytes // (1024 * 1024)}MB")

    def _load_cache_index(self) -> dict[str, dict[str, Any]]:
        if not self.cache_index_path.exists():
            return {}
        try:
            with open(self.cache_index_path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.warning(f"⚠️ Failed to load video cache index: {exc}")
        return {}

    def _save_cache_index(self) -> None:
        try:
            with open(self.cache_index_path, "w") as f:
                json.dump(self._index, f, indent=2)
        except Exception as exc:
            logger.warning(f"⚠️ Failed to persist video cache index: {exc}")

    def _purge_if_stale(self, key: str) -> dict[str, Any] | None:
        entry = self._index.get(key)
        if not entry:
            return None

        raw_path = Path(entry.get("raw_path", ""))
        if not raw_path.exists():
            self._index.pop(key, None)
            self._save_cache_index()
            logger.debug("🧹 Removed missing cache artifact for key=%s", key)
            return None

        cached_at = entry.get("cached_at")
        if cached_at:
            try:
                cached_dt = datetime.fromisoformat(cached_at)
                age_days = (datetime.now(UTC) - cached_dt).days
                if age_days > CACHE_EXPIRY_DAYS:
                    logger.info("🗑️ Cache entry expired key=%s age_days=%s", key, age_days)
                    try:
                        raw_path.unlink(missing_ok=True)
                    except Exception as exc:
                        logger.debug(
                            "⚠️ Failed to delete expired cache file %s: %s",
                            raw_path,
                            exc,
                        )
                    self._index.pop(key, None)
                    self._save_cache_index()
                    return None
            except Exception as exc:
                logger.debug("⚠️ Cache entry parse failed key=%s err=%s", key, exc)
        return entry

    @staticmethod
    def _is_supported_url(url: str) -> bool:
        try:
            base_url = url.split("#", 1)[0]
        except Exception:
            base_url = url
        return any(re.match(pattern, base_url) for pattern in SUPPORTED_PATTERNS)

    @staticmethod
    def _is_supported_instagram_content_path(path: str) -> bool:
        return bool(re.match(r"^/(?:p|reel|tv)/[0-9A-Za-z_-]+/?$", path or "") or re.match(r"^/stories/[^/]+/\d+/?$", path or ""))

    @staticmethod
    def _canonicalize_instagram_url_for_ytdlp(url: str) -> str:
        """Map supported kkinstagram mirror URLs to instagram.com for yt-dlp.
        d.vxinstagram pages expose direct media and are resolved separately to avoid Instagram login walls.
        Preserves path and query, drops fragments, and leaves unsupported paths/hosts unchanged.
        """
        if not url:
            return url
        try:
            parsed = urlparse(url)
            host = (parsed.netloc or "").lower()
            if host not in {"kkinstagram.com", "www.kkinstagram.com"}:
                return url.split("#", 1)[0]

            if not VideoIngestionManager._is_supported_instagram_content_path(parsed.path):
                return url

            canonical = parsed._replace(
                scheme="https",
                netloc="www.instagram.com",
                fragment="",
            )
            return urlunparse(canonical)
        except Exception:
            return url

    @staticmethod
    def _is_vxinstagram_page_url(url: str) -> bool:
        try:
            parsed = urlparse(url)
            return (parsed.netloc or "").lower() == "d.vxinstagram.com" and VideoIngestionManager._is_supported_instagram_content_path(parsed.path)
        except Exception:
            return False

    @staticmethod
    def _extract_vxinstagram_direct_media_url(html_text: str) -> str | None:
        for pattern in (
            r'<meta[^>]+property=["\']og:video(?::secure_url)?["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:video(?::secure_url)?["\']',
            r'<meta[^>]+name=["\']twitter:player:stream["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:player:stream["\']',
        ):
            match = re.search(pattern, html_text, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = html.unescape(match.group(1).strip())
            parsed = urlparse(candidate)
            if parsed.scheme in {"http", "https"} and (parsed.netloc or "").lower() == "d.vxinstagram.com" and VideoIngestionManager._is_direct_media_url(candidate):
                return candidate
        return None

    async def _resolve_vxinstagram_direct_media_url(self, url: str, timeout_s: float) -> str | None:
        if not self._is_vxinstagram_page_url(url):
            return None

        def _worker() -> str | None:
            req = urllib.request.Request(url.split("#", 1)[0], method="GET")
            req.add_header(
                "User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # nosec B310
                content_type = (resp.headers.get("Content-Type") or "").lower()
                if "text/html" not in content_type:
                    return None
                body = resp.read(256 * 1024).decode("utf-8", errors="ignore")
            return self._extract_vxinstagram_direct_media_url(body)

        try:
            media_url = await asyncio.to_thread(_worker)
        except Exception as exc:
            logger.debug("vxinstagram direct media resolution failed: %s", exc)
            return None

        if media_url:
            logger.info(
                "stt.vxinstagram.direct_media_resolved original_url=%s media_url=%s",
                url[:80],
                media_url[:120],
            )
        return media_url

    @staticmethod
    def _get_source_type(url: str) -> str:
        if "youtube.com" in url or "youtu.be" in url:
            return "youtube"
        if "tiktok.com" in url:
            return "tiktok"
        if any(
            host in url
            for host in (
                "instagram.com",
                "kkinstagram.com",
                "vxinstagram.com",
            )
        ):
            return "instagram"
        return "unknown"

    @staticmethod
    def _ext_rank(ext: str) -> int:
        preference = ["opus", "webm", "m4a", "mp3", "aac"]
        try:
            return preference.index((ext or "").lower())
        except ValueError:
            return len(preference)

    def _select_audio_format(self, metadata: dict[str, Any], url: str) -> tuple[dict[str, Any], bool]:
        formats = metadata.get("requested_downloads") or metadata.get("formats") or []
        audio_formats = []
        muxed_formats = []
        for fmt in formats:
            vcodec = (fmt.get("vcodec") or "").lower()
            acodec = (fmt.get("acodec") or "").lower()
            format_id = (fmt.get("format_id") or "").lower()
            # Skip if explicitly no audio codec AND not an audio-only format
            # HLS audio streams may have acodec=None but vcodec='none' and 'audio' in format_id [REH]
            is_audio_hint = "audio" in format_id or vcodec in ("", "none")
            if acodec == "none" and not is_audio_hint:
                continue
            if not acodec and not is_audio_hint:
                continue
            if vcodec in ("", "none"):
                audio_formats.append(fmt)
            elif acodec and acodec != "none":
                muxed_formats.append(fmt)

        def _key(fmt: dict[str, Any]) -> tuple:
            abr = fmt.get("abr") or fmt.get("tbr") or float("inf")
            if isinstance(abr, str):
                try:
                    abr = float(abr)
                except Exception:
                    abr = float("inf")
            abr_pref_penalty = 0 if abr <= 96 else abr
            size = fmt.get("filesize") or fmt.get("filesize_approx") or float("inf")
            return (
                abr_pref_penalty,
                self._ext_rank(fmt.get("ext") or ""),
                abr,
                size,
            )

        if audio_formats:
            selected_audio = min(audio_formats, key=_key)
            return selected_audio, False

        if not muxed_formats:
            msg = f"No audio-capable formats available for URL: {url}"
            raise VideoIngestError(msg)

        def _mux_key(fmt: dict[str, Any]) -> tuple:
            abr = fmt.get("abr") or fmt.get("tbr") or float("inf")
            if isinstance(abr, str):
                try:
                    abr = float(abr)
                except Exception:
                    abr = float("inf")
            abr_pref_penalty = 0 if abr <= 128 else abr
            height = fmt.get("height")
            try:
                height_val = int(height) if height is not None else 0
            except Exception:
                height_val = 0
            size = fmt.get("filesize") or fmt.get("filesize_approx") or float("inf")
            return (
                abr_pref_penalty,
                height_val,
                size,
                self._ext_rank(fmt.get("ext") or ""),
            )

        selected_mux = min(muxed_formats, key=_mux_key)
        return selected_mux, True

    @staticmethod
    def _hash_resolved_url(resolved_url: str) -> str:
        h = hashlib.sha256((resolved_url or "").encode("utf-8")).hexdigest()
        return h[:16]

    @staticmethod
    def _normalize_tiktok_url(url: str) -> str:
        """Normalize TikTok URLs to a canonical form for consistent cache keying.
        Strips tracking params and normalizes scheme/host variations.
        Also extracts video ID from player/embed URLs to map them to canonical identity.
        [REH][CA].
        """
        if not url:
            return url
        try:
            parsed = urlparse(url)
            # Normalize TikTok host variations
            host = parsed.netloc.lower()
            if host in (
                "vm.tiktok.com",
                "m.tiktok.com",
                "www.tiktok.com",
                "tiktok.com",
            ):
                path = parsed.path.rstrip("/")

                # Handle /player/v1/<video_id> embed URLs - extract video ID for identity
                player_match = re.match(r"^/player(?:/v\d+)?/(\d+)", path)
                if player_match:
                    video_id = player_match.group(1)
                    return f"tiktok://video/{video_id}"

                # Handle /@user/video/<video_id> canonical URLs - extract video ID
                video_match = re.match(r"^/@[\w\.-]+/video/(\d+)", path)
                if video_match:
                    video_id = video_match.group(1)
                    return f"tiktok://video/{video_id}"

                # For short URLs like /t/ZP8UxRTSU, the path is the key
                return f"tiktok://{path}"
        except Exception as exc:
            logger.debug(f"tiktok URL normalization failed: {exc}")
        return url

    @staticmethod
    def _is_tiktok_player_url(url: str) -> bool:
        """Check if URL is a TikTok player/embed URL that yt-dlp cannot handle directly.
        These URLs should be skipped for STT or deduplicated against the canonical URL.
        [REH][IV].
        """
        if not url:
            return False
        try:
            parsed = urlparse(url)
            host = parsed.netloc.lower()
            if host in (
                "vm.tiktok.com",
                "m.tiktok.com",
                "www.tiktok.com",
                "tiktok.com",
            ):
                path = parsed.path or ""
                # /player/ or /player/v1/ URLs are embed URLs
                if path.startswith("/player"):
                    return True
        except Exception as exc:
            logger.debug(f"tiktok player URL check failed: {exc}")
        return False

    @staticmethod
    def _normalize_youtube_url(url: str) -> str:
        """Normalize YouTube URLs to a canonical form for consistent cache keying.
        Extracts video ID from all URL variants: watch, shorts, embed, live, youtu.be.
        Returns canonical form: youtube://video/{VIDEO_ID}
        [REH][CA].
        """
        if not url:
            return url
        try:
            parsed = urlparse(url)
            host = parsed.netloc.lower()
            path = parsed.path or ""

            # youtu.be/VIDEO_ID
            if host in ("youtu.be", "www.youtu.be"):
                video_id = path.lstrip("/").split("/")[0].split("?")[0]
                if video_id and len(video_id) >= 6:
                    return f"youtube://video/{video_id}"

            # youtube.com variants
            if host in ("youtube.com", "www.youtube.com", "m.youtube.com"):
                # /watch?v=VIDEO_ID
                if path.startswith("/watch"):
                    from urllib.parse import parse_qs

                    query = parse_qs(parsed.query)
                    video_id = query.get("v", [""])[0]
                    if video_id and len(video_id) >= 6:
                        return f"youtube://video/{video_id}"

                # /shorts/VIDEO_ID, /embed/VIDEO_ID, /live/VIDEO_ID, /v/VIDEO_ID
                for prefix in ("/shorts/", "/embed/", "/live/", "/v/"):
                    if path.startswith(prefix):
                        video_id = path[len(prefix) :].split("/")[0].split("?")[0]
                        if video_id and len(video_id) >= 6:
                            return f"youtube://video/{video_id}"
        except Exception as exc:
            logger.debug(f"youtube URL normalization failed: {exc}")
        return url

    @staticmethod
    def _get_expected_extractor(url: str) -> str | None:
        """Get expected yt-dlp extractor for a URL based on domain.
        Returns None if domain is unknown (allows generic extractor).
        [IV].
        """
        try:
            parsed = urlparse(url)
            host = parsed.netloc.lower()
            return _DOMAIN_EXTRACTOR_MAP.get(host)
        except Exception as exc:
            logger.debug(f"extractor lookup failed: {exc}")
            return None

    @staticmethod
    def _canonicalize_video_identity(
        original_url: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Canonicalize video identity for cache keying across all providers.
        Uses yt-dlp metadata (extractor + id) when available, falls back to
        provider-specific URL normalization for known domains.
        [REH][CA].
        """
        # If we have yt-dlp metadata, use extractor:id as the canonical identity
        if metadata:
            extractor = metadata.get("extractor_key") or metadata.get("extractor") or ""
            video_id = metadata.get("id") or ""
            if extractor and video_id:
                return f"{extractor.lower()}:{video_id}"

        # Fallback: use provider-specific URL normalization
        if not original_url:
            return ""

        url_lower = original_url.lower()

        # YouTube normalization
        if "youtube.com" in url_lower or "youtu.be" in url_lower:
            normalized = VideoIngestionManager._normalize_youtube_url(original_url)
            if normalized.startswith("youtube://"):
                return normalized.replace("://", ":")

        # TikTok normalization
        if "tiktok.com" in url_lower:
            normalized = VideoIngestionManager._normalize_tiktok_url(original_url)
            if normalized.startswith("tiktok://"):
                return normalized.replace("://", ":")

        # Generic fallback: hash of original URL
        return f"generic:{hashlib.sha256(original_url.encode()).hexdigest()[:16]}"

    def _compute_download_key(
        self,
        resolved_url: str,
        fmt_id: str,
        content_length: int | None,
        original_url: str | None = None,
        video_identity: str | None = None,
    ) -> str:
        """Compute a unique cache key for a download job.

        ALWAYS includes video identity to prevent cross-contamination across
        different videos that may share CDN URLs or similar resolved paths.
        [REH][CA]
        """
        length_part = str(content_length) if content_length is not None else "na"
        base_key = f"{self._hash_resolved_url(resolved_url)}-{fmt_id}-{length_part}"

        # Always include video identity hash to prevent cross-contamination [REH]
        if video_identity:
            identity_hash = self._hash_resolved_url(video_identity)[:10]
            base_key = f"{base_key}-v{identity_hash}"
        elif original_url:
            # Fallback: compute identity from original URL if not provided
            fallback_identity = self._canonicalize_video_identity(original_url)
            identity_hash = self._hash_resolved_url(fallback_identity)[:10]
            base_key = f"{base_key}-v{identity_hash}"

        return base_key

    def _get_cache_entry(self, download_key: str) -> tuple[dict[str, Any], Path] | None:
        entry = self._purge_if_stale(download_key)
        if not entry:
            return None
        raw_path = Path(entry.get("raw_path"))
        if not raw_path.exists():
            return None
        return entry, raw_path

    def _should_apply_cookies(self, url: str) -> bool:
        source = self._get_source_type(url)
        return bool(YTDLP_COOKIES_FROM_BROWSER or YTDLP_COOKIES_FILE) and (not YTDLP_COOKIES_SITES or source in YTDLP_COOKIES_SITES)

    def _augment_with_cookies(self, cmd: list, url: str) -> list:
        cmd = list(cmd)
        if not self._should_apply_cookies(url):
            return cmd
        if YTDLP_COOKIES_FROM_BROWSER:
            cmd += ["--cookies-from-browser", YTDLP_COOKIES_FROM_BROWSER]
            logger.debug("🔑 Applying browser cookies for yt-dlp")
        elif YTDLP_COOKIES_FILE:
            cmd += ["--cookies", YTDLP_COOKIES_FILE]
            logger.debug("🔑 Applying cookies file for yt-dlp")
        return cmd

    async def _run_subprocess(self, cmd: list, timeout_s: float, label: str) -> tuple[bytes, bytes]:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except TimeoutError:
            with contextlib.suppress(Exception):
                proc.kill()
            logger.exception("stt.fail reason=download_timeout stage=%s", label)
            msg = f"{label} timed out after {timeout_s:.0f}s"
            raise VideoIngestError(msg)
        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            msg = f"{label} failed: {err or 'unknown error'}"
            raise VideoIngestError(msg)
        return stdout, stderr

    @staticmethod
    def _is_direct_media_url(url: str) -> bool:
        parsed = urlparse(url)
        suffix = Path(parsed.path).suffix.lower()
        return suffix in {".mp4", ".m4a", ".aac", ".opus", ".ogg", ".mp3", ".webm"}

    async def _download_direct_media(
        self,
        url: str,
        ext: str,
        timeout_s: float,
    ) -> tuple[Path, int | None]:
        # Validate URL scheme and block private/internal IPs (SSRF protection)
        _ssrf_parsed = urlparse(url)
        if _ssrf_parsed.scheme not in ("http", "https"):
            msg = f"Unsupported URL scheme: {_ssrf_parsed.scheme}"
            raise VideoIngestError(msg)
        _ssrf_hostname = _ssrf_parsed.hostname or ""
        if _is_private_hostname(_ssrf_hostname):
            msg = f"SSRF blocked: {_ssrf_hostname}"
            raise VideoIngestError(msg)

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            temp_path = Path(tmp.name)

        def _worker() -> int | None:
            req = urllib.request.Request(url, method="GET")
            req.add_header(
                "User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # nosec B310
                content_length = resp.headers.get("Content-Length")
                with open(temp_path, "wb") as fh:
                    shutil.copyfileobj(resp, fh)
            try:
                return int(content_length) if content_length else None
            except Exception:
                return None

        try:
            content_length = await asyncio.to_thread(_worker)
        except Exception as exc:
            with contextlib.suppress(Exception):
                temp_path.unlink(missing_ok=True)
            msg = f"Direct media download failed: {exc}"
            raise VideoIngestError(msg) from exc

        return temp_path, content_length

    async def _probe_metadata(self, url: str, timeout_s: float) -> dict[str, Any]:
        # Log the exact URL being probed for STT debugging [REH]
        logger.info(
            "stt.ytdlp.probe url=%s",
            url[:120] if url else "none",
        )
        cmd = ["yt-dlp", "--dump-json", "--no-playlist", "--quiet", url]
        cmd = self._augment_with_cookies(cmd, url)
        stdout, _ = await self._run_subprocess(cmd, timeout_s, "yt-dlp metadata probe")
        try:
            metadata = json.loads(stdout.decode())
        except json.JSONDecodeError as exc:
            msg = f"Failed to parse yt-dlp metadata: {exc}"
            raise VideoIngestError(msg)

        # Log the resolved URL from yt-dlp for debugging [REH]
        resolved_id = metadata.get("id") or "unknown"
        resolved_webpage = metadata.get("webpage_url") or metadata.get("url") or "none"
        logger.info(
            "stt.ytdlp.resolved id=%s webpage_url=%s",
            resolved_id[:40] if resolved_id else "none",
            resolved_webpage[:80] if resolved_webpage else "none",
        )
        return metadata

    async def _download_audio(
        self,
        source_url: str,
        format_id: str,
        ext: str,
        output_dir: Path,
        timeout_s: float,
    ) -> Path:
        # Log the exact URL being downloaded for STT debugging [REH]
        logger.info(
            "stt.ytdlp.download url=%s format=%s",
            source_url[:120] if source_url else "none",
            format_id,
        )
        out_template = output_dir / "%(id)s.%(ext)s"
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            "--no-progress",
            "--concurrent-fragments",
            "1",
            "--retries",
            "1",
            "--fragment-retries",
            "1",
            "--retry-sleep",
            "1",
            "--socket-timeout",
            "10",
            "--format",
            format_id,
            "--output",
            str(out_template),
            "--paths",
            f"temp:{output_dir}",
            "--no-part",
            "--print",
            "after_move:filepath",
            source_url,
        ]
        cmd = self._augment_with_cookies(cmd, source_url)
        stdout, _ = await self._run_subprocess(cmd, timeout_s, "yt-dlp download")
        filepath = stdout.decode().strip()
        if not filepath:
            msg = "yt-dlp did not emit output filepath"
            raise VideoIngestError(msg)
        path = Path(filepath)
        if not path.exists():
            msg = f"Downloaded file missing: {filepath}"
            raise VideoIngestError(msg)
        if path.suffix.lower() != f".{ext.lower()}":
            # Ensure suffix matches expectation for consistent caching
            target = path.with_suffix(f".{ext.lower()}")
            try:
                path.rename(target)
                path = target
            except Exception as exc:
                logger.debug(f"suffix rename failed: {exc}")
        return path

    async def fetch_and_prepare_url_audio(self, url: str, force_refresh: bool = False) -> DownloadedAudio:
        """Fetch raw audio for URL via yt-dlp using audio-only formats.

        Returns a DownloadedAudio artifact that downstream preprocessing can consume.
        """
        if not self._is_supported_url(url):
            msg = f"Unsupported URL format: {url}"
            raise VideoIngestError(msg)

        async with _download_semaphore:
            metadata_timeout = float(os.getenv("YTDLP_METADATA_TIMEOUT_S", "10"))
            download_timeout = float(os.getenv("YTDLP_DOWNLOAD_TIMEOUT_S", "25"))
            # Use live config so hot-reloads via reload_env take effect [REH][PA]
            cfg = load_config()
            budget_limit_cfg = cfg.get("MEDIA_PER_ITEM_BUDGET")
            if budget_limit_cfg is not None:
                try:
                    budget_s = max(15.0, float(budget_limit_cfg))
                    metadata_timeout = min(metadata_timeout, max(5.0, budget_s * 0.25))
                    download_timeout = min(download_timeout, max(15.0, budget_s - 5.0))
                except Exception as exc:
                    logger.debug(f"budget timeout calc failed: {exc}")

            vx_direct_media_url = await self._resolve_vxinstagram_direct_media_url(url, min(metadata_timeout, 10.0))
            if vx_direct_media_url:
                parsed_direct = urlparse(vx_direct_media_url)
                return await self._direct_media_fallback(
                    original_url=url,
                    media_url=vx_direct_media_url,
                    parsed=parsed_direct,
                    force_refresh=force_refresh,
                    timeout_s=download_timeout,
                )

            url_no_fragment = self._canonicalize_instagram_url_for_ytdlp(url)
            if url_no_fragment != url.split("#", 1)[0]:
                logger.info(
                    "stt.instagram.canonicalized original_host=%s canonical_url=%s",
                    (urlparse(url).netloc or "").lower(),
                    url_no_fragment[:120],
                )
            parsed_url = urlparse(url_no_fragment)
            direct_candidate = self._is_direct_media_url(url_no_fragment)

            demux_required = False
            try:
                metadata = await self._probe_metadata(url_no_fragment, metadata_timeout)
                selected, demux_required = self._select_audio_format(metadata, url_no_fragment)
            except VideoIngestError as exc:
                if direct_candidate or "NumericString value expected" in str(exc):
                    logger.info(
                        "yt-dlp metadata failed (%s); attempting direct media fallback",
                        exc,
                    )
                    return await self._direct_media_fallback(
                        original_url=url,
                        media_url=url_no_fragment,
                        parsed=parsed_url,
                        force_refresh=force_refresh,
                        timeout_s=download_timeout,
                    )
                raise

            # Validate yt-dlp metadata matches expected provider for known domains [REH][IV]
            expected_extractor = self._get_expected_extractor(url_no_fragment)
            actual_extractor = (metadata.get("extractor_key") or metadata.get("extractor") or "").lower()
            webpage_url = metadata.get("webpage_url") or ""
            metadata_video_id = metadata.get("id") or ""

            if expected_extractor and actual_extractor:
                # Check for obvious mismatches (e.g., YouTube URL returning TikTok extractor)
                if expected_extractor != actual_extractor:
                    logger.warning(
                        "stt.ytdlp.mismatch expected=%s got=%s webpage_url=%s original=%s",
                        expected_extractor,
                        actual_extractor,
                        webpage_url[:80] if webpage_url else "none",
                        url_no_fragment[:80],
                    )
                    msg = f"yt-dlp returned unexpected content: expected {expected_extractor}, got {actual_extractor}"
                    raise VideoIngestError(msg)

            # Additional validation: verify video ID from URL matches metadata ID [REH][IV]
            # This prevents cross-contamination when yt-dlp resolves to wrong video
            if expected_extractor == "youtube":
                normalized = self._normalize_youtube_url(url_no_fragment)
                if normalized.startswith("youtube://video/"):
                    url_video_id = normalized.split("/")[-1]
                    if metadata_video_id and url_video_id and url_video_id != metadata_video_id:
                        logger.warning(
                            "stt.ytdlp.id_mismatch url_id=%s metadata_id=%s url=%s",
                            url_video_id,
                            metadata_video_id,
                            url_no_fragment[:60],
                        )
                        msg = f"Video ID mismatch: URL suggests {url_video_id} but yt-dlp returned {metadata_video_id}"
                        raise VideoIngestError(msg)
            elif expected_extractor == "tiktok":
                normalized = self._normalize_tiktok_url(url_no_fragment)
                if normalized.startswith("tiktok://video/"):
                    url_video_id = normalized.split("/")[-1]
                    if metadata_video_id and url_video_id and url_video_id != metadata_video_id:
                        logger.warning(
                            "stt.ytdlp.id_mismatch url_id=%s metadata_id=%s url=%s",
                            url_video_id,
                            metadata_video_id,
                            url_no_fragment[:60],
                        )
                        msg = f"Video ID mismatch: URL suggests {url_video_id} but yt-dlp returned {metadata_video_id}"
                        raise VideoIngestError(msg)

            # Compute canonical video identity for cache keying [REH]
            video_identity = self._canonicalize_video_identity(url, metadata)
            logger.debug(
                "stt.video.identity original=%s identity=%s extractor=%s id=%s",
                url[:60] if url else "none",
                video_identity[:40] if video_identity else "none",
                actual_extractor,
                metadata.get("id", "none")[:20],
            )

            resolved_url = selected.get("url") or metadata.get("url") or url_no_fragment
            fmt_id = str(selected.get("format_id") or selected.get("format"))
            ext = (selected.get("ext") or "m4a").lower()
            duration = float(metadata.get("duration") or 0.0)
            if duration and duration > MAX_DURATION_SECONDS:
                msg = f"Video too long: {duration:.1f}s (max {MAX_DURATION_SECONDS}s)"
                raise VideoIngestError(msg)
            title = metadata.get("title", "Unknown Title")
            uploader = metadata.get("uploader", "Unknown")
            upload_date = metadata.get("upload_date", "")
            content_length = selected.get("filesize") or selected.get("filesize_approx")
            try:
                if isinstance(content_length, str):
                    content_length = float(content_length)
            except Exception:
                content_length = None
            if isinstance(content_length, float):
                content_length = int(content_length)

            if content_length and content_length > self._size_guard_bytes:
                msg = f"Audio payload too large: {content_length} bytes (limit {self._size_guard_bytes})"
                raise VideoIngestError(msg)

            download_key = self._compute_download_key(
                resolved_url,
                fmt_id,
                content_length,
                original_url=url,
                video_identity=video_identity,
            )
            cache_entry = None if force_refresh else self._get_cache_entry(download_key)

            # Log cache lookup for STT debugging [REH]
            logger.info(
                "stt.identity original_url=%s canonical=%s extractor=%s video_id=%s cache_key=%s",
                url[:60] if url else "none",
                video_identity[:40] if video_identity else "none",
                actual_extractor,
                metadata.get("id", "none")[:20],
                download_key[:20],
            )
            logger.debug(
                "stt.cache.lookup key=%s original_url=%s resolved_url=%s",
                download_key[:20],
                url[:60] if url else "none",
                resolved_url[:60] if resolved_url else "none",
            )

            cache_hit = False
            if cache_entry:
                entry, raw_path = cache_entry
                stat_size = raw_path.stat().st_size
                expected = entry.get("content_length")
                if expected and abs(stat_size - expected) > max(65536, expected * 0.1):
                    logger.debug(
                        "⚠️ Cached artifact size mismatch key=%s expected=%s actual=%s",
                        download_key,
                        expected,
                        stat_size,
                    )
                else:
                    cache_hit = True
                    logger.info(
                        "cache.hit stage=download key=%s ext=%s size=%s",
                        download_key[:12],
                        ext,
                        stat_size,
                    )
                    return DownloadedAudio(
                        raw_path=raw_path,
                        metadata=VideoMetadata(
                            url=url,
                            title=title,
                            duration_seconds=duration,
                            uploader=uploader,
                            upload_date=upload_date,
                            source_type=self._get_source_type(url),
                        ),
                        download_key=download_key,
                        format_id=fmt_id,
                        resolved_url=resolved_url,
                        content_length=content_length,
                        cache_hit=True,
                        ext=ext,
                        timestamp=datetime.now(UTC),
                        demux_fallback=demux_required,
                    )

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                attempts = 2
                for attempt in range(1, attempts + 1):
                    try:
                        raw_download = await self._download_audio(
                            url_no_fragment,
                            fmt_id,
                            ext,
                            temp_path,
                            download_timeout,
                        )
                        break
                    except VideoIngestError as exc:
                        logger.warning(
                            "⚠️ yt-dlp attempt %s/%s failed: %s",
                            attempt,
                            attempts,
                            exc,
                        )
                        if attempt == attempts:
                            logger.exception(
                                "stt.fail reason=download_timeout stage=yt-dlp retries=%s",
                                attempts,
                            )
                            msg = "Failed to download audio after retries"
                            raise VideoIngestError(msg) from exc
                else:
                    msg = "yt-dlp retry loop exhausted"
                    raise VideoIngestError(msg)

                raw_cache_path = self.cache_dir / f"{download_key}.{ext}"
                raw_cache_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(str(raw_download), raw_cache_path)
                except Exception as exc:
                    logger.warning(
                        "⚠️ Failed to move download into cache (%s → %s): %s",
                        raw_download,
                        raw_cache_path,
                        exc,
                    )
                    try:
                        shutil.copy2(str(raw_download), raw_cache_path)
                    except Exception as copy_exc:
                        msg = f"Failed to persist downloaded audio: {copy_exc}"
                        raise VideoIngestError(msg) from copy_exc

            stat_size = raw_cache_path.stat().st_size if raw_cache_path.exists() else 0
            self._index[download_key] = {
                "raw_path": str(raw_cache_path),
                "content_length": content_length,
                "format_id": fmt_id,
                "ext": ext,
                "source_url": url,
                "cached_at": datetime.now(UTC).isoformat(),
                "demux_fallback": demux_required,
            }
            self._save_cache_index()
            logger.info(
                "cache.store stage=download key=%s ext=%s size=%s",
                download_key[:12],
                ext,
                stat_size,
            )

            metadata_obj = VideoMetadata(
                url=url,
                title=title,
                duration_seconds=duration,
                uploader=uploader,
                upload_date=upload_date,
                source_type=self._get_source_type(url),
            )

            return DownloadedAudio(
                raw_path=raw_cache_path,
                metadata=metadata_obj,
                download_key=download_key,
                format_id=fmt_id,
                resolved_url=resolved_url,
                content_length=content_length,
                cache_hit=cache_hit,
                ext=ext,
                timestamp=datetime.now(UTC),
                demux_fallback=demux_required,
            )

    async def _direct_media_fallback(
        self,
        original_url: str,
        media_url: str,
        parsed: ParseResult,
        force_refresh: bool,
        timeout_s: float,
    ) -> DownloadedAudio:
        ext = Path(parsed.path).suffix.lstrip(".") or "mp4"
        audio_exts = {"aac", "m4a", "mp3", "opus", "ogg", "flac", "wav"}
        demux_flag = ext.lower() not in audio_exts
        download_key = f"{self._hash_resolved_url(media_url)}-direct"
        if not force_refresh:
            cache_entry = self._get_cache_entry(download_key)
            if cache_entry:
                entry, raw_path = cache_entry
                stat_size = raw_path.stat().st_size
                logger.info(
                    "cache.hit stage=download key=%s ext=%s size=%s",
                    download_key[:12],
                    ext,
                    stat_size,
                )
                metadata_obj = VideoMetadata(
                    url=original_url,
                    title=Path(parsed.path).name or "direct-media",
                    duration_seconds=float(entry.get("duration_seconds", 0.0) or 0.0),
                    uploader="",
                    upload_date="",
                    source_type=self._get_source_type(original_url),
                )
                return DownloadedAudio(
                    raw_path=raw_path,
                    metadata=metadata_obj,
                    download_key=download_key,
                    format_id="direct",
                    resolved_url=media_url,
                    content_length=entry.get("content_length"),
                    cache_hit=True,
                    ext=ext,
                    timestamp=datetime.now(UTC),
                    demux_fallback=demux_flag,
                )

        temp_path, content_length = await self._download_direct_media(media_url, ext, timeout_s)

        raw_cache_path = self.cache_dir / f"{download_key}.{ext}"
        raw_cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(temp_path), raw_cache_path)
        except Exception:
            shutil.copy2(str(temp_path), raw_cache_path)
            Path(temp_path).unlink(missing_ok=True)

        stat_size = raw_cache_path.stat().st_size if raw_cache_path.exists() else 0
        metadata_obj = VideoMetadata(
            url=original_url,
            title=Path(parsed.path).name or "direct-media",
            duration_seconds=0.0,
            uploader="",
            upload_date="",
            source_type=self._get_source_type(original_url),
        )
        self._index[download_key] = {
            "raw_path": str(raw_cache_path),
            "content_length": content_length or stat_size,
            "format_id": "direct",
            "ext": ext,
            "source_url": original_url,
            "cached_at": datetime.now(UTC).isoformat(),
            "duration_seconds": 0.0,
            "demux_fallback": demux_flag,
        }
        self._save_cache_index()
        logger.info(
            "cache.store stage=download key=%s ext=%s size=%s",
            download_key[:12],
            ext,
            stat_size,
        )

        return DownloadedAudio(
            raw_path=raw_cache_path,
            metadata=metadata_obj,
            download_key=download_key,
            format_id="direct",
            resolved_url=media_url,
            content_length=content_length or stat_size,
            cache_hit=False,
            ext=ext,
            timestamp=datetime.now(UTC),
            demux_fallback=demux_flag,
        )


# Global instance
video_manager = VideoIngestionManager()


async def fetch_and_prepare_url_audio(url: str, force_refresh: bool = False) -> DownloadedAudio:
    """Convenience wrapper returning a DownloadedAudio artifact."""
    return await video_manager.fetch_and_prepare_url_audio(url, force_refresh)
