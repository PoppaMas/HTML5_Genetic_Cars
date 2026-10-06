"""Socket I/O declared by stock models (737: telnet 5137, QTJSBSIM UDP 5139) is stripped without changing physics."""
import os

from evolution import genome, sim


def test_737_sanitized_copy_has_no_sockets_and_same_result():
    root = sim._aircraft_root("737")
    assert root and os.path.exists(os.path.join(root, "737", "737.xml"))
    txt = open(os.path.join(root, "737", "737.xml")).read()
    assert "port=" not in txt
    assert sim._aircraft_root("c172x") is None  # c172x's telnet input is commented out: stock path
    P = sim.Profile(aircraft="737", h0_ft=10000, speed_kts=250, duration_s=20)
    sc = sim.make_scenarios(2, 7, P)[1]
    g = genome.decode([0.6] * genome.N_GENES)
    a = sim.simulate(g, sc, P)
    saved = sim._ROOT_MEMO[(None, "737")]
    try:
        sim._ROOT_MEMO[(None, "737")] = None  # force the stock (socket-declaring) model
        b = sim.simulate(g, sc, P)
    finally:
        sim._ROOT_MEMO[(None, "737")] = saved
    assert a["cost"] == b["cost"] and a["status"] == b["status"]


def test_strip_sockets_keeps_fcs_inputs():
    x = ('<fdm_config><!-- <input port="1"> --><flight_control><channel><summer><input>fcs/a</input>'
         '<output>fcs/b</output></summer></channel></flight_control><input port="5137"/>'
         '<input port="5139" type="QTJSBSIM" rate="20"><property>fcs/elevator-cmd-norm</property></input>'
         '<output name="x" type="SOCKET" port="1138" rate="10"><property>a</property></output></fdm_config>')
    y = sim.strip_sockets(x)
    assert "port" not in y and "<input>fcs/a</input>" in y and "<output>fcs/b</output>" in y
