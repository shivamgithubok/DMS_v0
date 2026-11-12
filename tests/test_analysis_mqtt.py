from dms import alerts as A
from dms.config import load_config
from dms.mqtt_publisher import MqttPublisher
from dms.state import SharedState
from dms.vlm_worker import ChannelAnalysis, analysis_payload


def test_analysis_topics_and_queue(tmp_path):
    cfg = load_config(None, {"device_id": "t1", "data_dir": str(tmp_path), "mqtt": {"heartbeat_s": 60}})
    shared, bus = SharedState(), A.AlertBus("t1", cooldown_s=0)
    pub = MqttPublisher(cfg, bus, shared)
    try:
        assert pub.topic_dms_analysis == "dms/t1/analysis"
        assert pub.topic_cms_analysis == "cms/t1/analysis"
        dms = analysis_payload("t1", "dms", ChannelAnalysis(scene="hands on wheel"), 1)
        cms = analysis_payload("t1", "cms", ChannelAnalysis(scene="cabin calm", kept=["child_unattended"]), 1)
        bus.emit_analysis("dms", dms)
        bus.emit_analysis("cms", cms)
        assert bus.analysis_queue.qsize() == 2
        pub._drain_analysis()
        assert pub._latest_analysis["dms"]["scene"] == "hands on wheel"
        assert pub._latest_analysis["cms"]["hazards"] == ["child_unattended"]
    finally:
        pub.outbox.close()
