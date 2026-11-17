import re
import xml.etree.ElementTree as ET

with open("arch_DRC_updatedv2.drawio.xml", "r", encoding="utf-8") as f:
    raw = f.read()

# Вытаскиваем чистый Base64 картинки, игнорируя любые сломанные теги вокруг
match = re.search(r"iVBORw0KGgo[A-Za-z0-9+/=\s]+", raw)
if not match:
    raise ValueError("Не нашёл PNG внутри arch_DRC_updatedv2.drawio.xml")

clean_b64 = re.sub(r"\s+", "", match.group(0))

final_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net">
  <diagram name="Страница-1" id="bI6O2f4mDUCClSYU1rP_">
    <mxGraphModel dx="1226" dy="803" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1169" pageHeight="827" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />

        <mxCell id="backbone-bg" parent="1" style="rounded=1;whiteSpace=wrap;html=1;fontFamily=Times New Roman;fillColor=#f8fafc;strokeColor=#cbd5e1;strokeWidth=1;dashed=1;dashPattern=4 4;verticalAlign=top;align=left;spacingLeft=10;spacingTop=6;fontSize=12;fontStyle=1;fontColor=#475569;arcSize=6;" value="Spatial Feature Extractor &amp; Evidential Predictor" vertex="1">
          <mxGeometry height="156" width="400" x="150" y="108" as="geometry" />
        </mxCell>

        <mxCell id="input-img" parent="1" style="shape=image;verticalLabelPosition=bottom;labelBackgroundColor=none;verticalAlign=top;aspect=fixed;imageAspect=0;image=data:image/png,{clean_b64}" vertex="1">
          <mxGeometry height="84" width="84" x="36" y="148" as="geometry" />
        </mxCell>
        <mxCell id="input-lbl" parent="1" style="text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=center;verticalAlign=top;rounded=0;fontFamily=Times New Roman;fontSize=12;fontColor=#0f172a;" value="&lt;b&gt;Input Layout&lt;/b&gt;&lt;br&gt;&lt;i&gt;&lt;b&gt;X&lt;/b&gt;&lt;/i&gt; &amp;isin; &amp;#8477;&lt;sup&gt;&lt;i&gt;C&lt;/i&gt;&amp;times;&lt;i&gt;H&lt;/i&gt;&amp;times;&lt;i&gt;W&lt;/i&gt;&lt;/sup&gt;" vertex="1">
          <mxGeometry height="38" width="108" x="24" y="236" as="geometry" />
        </mxCell>

        <mxCell id="arr-in-e1" edge="1" parent="1" style="endArrow=block;endFill=1;endSize=5;html=1;rounded=0;strokeWidth=1.5;strokeColor=#475569;" value="">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="122" y="190" as="sourcePoint" />
            <mxPoint x="164" y="190" as="targetPoint" />
          </mxGeometry>
        </mxCell>

        <mxCell id="enc-1" parent="1" style="shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.06;darkOpacity2=0.14;size=9;fillColor=#dbeafe;strokeColor=#3b82f6;strokeWidth=1.1;fontFamily=Times New Roman;fontSize=11;" value="" vertex="1">
          <mxGeometry height="92" width="32" x="166" y="144" as="geometry" />
        </mxCell>
        <mxCell id="arr-e1-e2" edge="1" parent="1" style="endArrow=block;endFill=1;endSize=4;html=1;rounded=0;strokeWidth=1.4;strokeColor=#475569;" value="">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="199" y="190" as="sourcePoint" />
            <mxPoint x="223" y="190" as="targetPoint" />
          </mxGeometry>
        </mxCell>

        <mxCell id="enc-2" parent="1" style="shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.06;darkOpacity2=0.14;size=9;fillColor=#bfdbfe;strokeColor=#2563eb;strokeWidth=1.1;fontFamily=Times New Roman;fontSize=11;" value="" vertex="1">
          <mxGeometry height="74" width="40" x="225" y="153" as="geometry" />
        </mxCell>
        <mxCell id="arr-e2-e3" edge="1" parent="1" style="endArrow=block;endFill=1;endSize=4;html=1;rounded=0;strokeWidth=1.4;strokeColor=#475569;" value="">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="266" y="190" as="sourcePoint" />
            <mxPoint x="290" y="190" as="targetPoint" />
          </mxGeometry>
        </mxCell>

        <mxCell id="enc-3" parent="1" style="shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.06;darkOpacity2=0.14;size=9;fillColor=#93c5fd;strokeColor=#1d4ed8;strokeWidth=1.1;fontFamily=Times New Roman;fontSize=12;fontStyle=1;fontColor=#1e3a8a;" value="&lt;i&gt;F&lt;/i&gt;" vertex="1">
          <mxGeometry height="56" width="48" x="292" y="162" as="geometry" />
        </mxCell>

        <mxCell id="enc-lbl" parent="1" style="text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;rounded=0;fontFamily=Times New Roman;fontSize=11;fontColor=#1e3a8a;" value="Spatial Encoder (&lt;i&gt;F&lt;/i&gt; &amp;isin; &amp;#8477;&lt;sup&gt;&lt;i&gt;C&#39;&lt;/i&gt;&amp;times;&lt;i&gt;H&#39;&lt;/i&gt;&amp;times;&lt;i&gt;W&#39;&lt;/i&gt;&lt;/sup&gt;)" vertex="1">
          <mxGeometry height="22" width="184" x="160" y="238" as="geometry" />
        </mxCell>

        <mxCell id="arr-e3-mod" edge="1" parent="1" style="endArrow=block;endFill=1;endSize=4;html=1;rounded=0;strokeWidth=1.4;strokeColor=#475569;" value="">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="341" y="190" as="sourcePoint" />
            <mxPoint x="364" y="190" as="targetPoint" />
          </mxGeometry>
        </mxCell>

        <mxCell id="mod-node" parent="1" style="ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#fef3c7;strokeColor=#d97706;strokeWidth=1.3;fontFamily=Times New Roman;fontSize=12;fontStyle=1;fontColor=#92400e;" value="&amp;otimes;" vertex="1">
          <mxGeometry height="20" width="20" x="365" y="180" as="geometry" />
        </mxCell>

        <mxCell id="arr-mod-c1" edge="1" parent="1" style="endArrow=block;endFill=1;endSize=4;html=1;rounded=0;strokeWidth=1.4;strokeColor=#475569;" value="">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="386" y="190" as="sourcePoint" />
            <mxPoint x="408" y="190" as="targetPoint" />
          </mxGeometry>
        </mxCell>

        <mxCell id="cond-block" parent="1" style="shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.06;darkOpacity2=0.14;size=9;fillColor=#fde68a;strokeColor=#d97706;strokeWidth=1.1;fontFamily=Times New Roman;fontSize=12;fontStyle=1;fontColor=#78350f;" value="&lt;i&gt;F&lt;sub&gt;c&lt;/sub&gt;&lt;/i&gt;" vertex="1">
          <mxGeometry height="56" width="48" x="410" y="162" as="geometry" />
        </mxCell>
        <mxCell id="arr-c1-head" edge="1" parent="1" style="endArrow=block;endFill=1;endSize=4;html=1;rounded=0;strokeWidth=1.4;strokeColor=#475569;" value="">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="459" y="190" as="sourcePoint" />
            <mxPoint x="482" y="190" as="targetPoint" />
          </mxGeometry>
        </mxCell>

        <mxCell id="evid-head" parent="1" style="shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.06;darkOpacity2=0.14;size=9;fillColor=#e0e7ff;strokeColor=#4f46e5;strokeWidth=1.1;fontFamily=Times New Roman;fontSize=11;fontStyle=1;fontColor=#312e81;" value="Beta&lt;br&gt;Head" vertex="1">
          <mxGeometry height="56" width="54" x="484" y="162" as="geometry" />
        </mxCell>
        <mxCell id="head-lbl" parent="1" style="text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;rounded=0;fontFamily=Times New Roman;fontSize=11;fontColor=#312e81;" value="Evidence (&amp;alpha;&lt;sub&gt;&lt;i&gt;k&lt;/i&gt;&lt;/sub&gt;, &amp;beta;&lt;sub&gt;&lt;i&gt;k&lt;/i&gt;&lt;/sub&gt;)" vertex="1">
          <mxGeometry height="22" width="124" x="414" y="238" as="geometry" />
        </mxCell>

        <mxCell id="arr-out-top" edge="1" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=1.5;strokeColor=#475569;endArrow=block;endFill=1;endSize=5;" value="">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="560" y="190" />
              <mxPoint x="560" y="140" />
            </Array>
            <mxPoint x="539" y="190" as="sourcePoint" />
            <mxPoint x="580" y="140" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        <mxCell id="arr-out-bot" edge="1" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=1.5;strokeColor=#475569;endArrow=block;endFill=1;endSize=5;" value="">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="560" y="190" />
              <mxPoint x="560" y="236" />
            </Array>
            <mxPoint x="539" y="190" as="sourcePoint" />
            <mxPoint x="580" y="236" as="targetPoint" />
          </mxGeometry>
        </mxCell>

        <mxCell id="out-drc" parent="1" style="shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.06;darkOpacity2=0.14;size=8;fillColor=#fee2e2;strokeColor=#ef4444;strokeWidth=1.1;fontFamily=Times New Roman;fontSize=12;" value="" vertex="1">
          <mxGeometry height="48" width="48" x="582" y="116" as="geometry" />
        </mxCell>
        <mxCell id="out-drc-lbl" parent="1" style="text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;rounded=0;fontFamily=Times New Roman;fontSize=12;fontColor=#0f172a;" value="&lt;b&gt;DRC Hotspot Map&lt;/b&gt;&lt;br&gt;&lt;i&gt;&lt;b&gt;&amp;#374;&lt;/b&gt;&lt;/i&gt; &amp;isin; &amp;#8477;&lt;sup&gt;&lt;i&gt;K&lt;/i&gt;&amp;times;&lt;i&gt;H&#39;&lt;/i&gt;&amp;times;&lt;i&gt;W&#39;&lt;/i&gt;&lt;/sup&gt;" vertex="1">
          <mxGeometry height="44" width="128" x="636" y="118" as="geometry" />
        </mxCell>

        <mxCell id="out-unc" parent="1" style="shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.06;darkOpacity2=0.14;size=8;fillColor=#dcfce7;strokeColor=#22c55e;strokeWidth=1.1;fontFamily=Times New Roman;fontSize=12;" value="" vertex="1">
          <mxGeometry height="48" width="48" x="582" y="212" as="geometry" />
        </mxCell>
        <mxCell id="out-unc-lbl" parent="1" style="text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;rounded=0;fontFamily=Times New Roman;fontSize=12;fontColor=#0f172a;" value="&lt;b&gt;Uncertainty Map&lt;/b&gt;&lt;br&gt;&lt;i&gt;&lt;b&gt;U&lt;/b&gt;&lt;/i&gt; &amp;isin; [0, 1]&lt;sup&gt;&lt;i&gt;K&lt;/i&gt;&amp;times;&lt;i&gt;H&#39;&lt;/i&gt;&amp;times;&lt;i&gt;W&#39;&lt;/i&gt;&lt;/sup&gt;" vertex="1">
          <mxGeometry height="44" width="128" x="636" y="214" as="geometry" />
        </mxCell>

        <mxCell id="p-1" parent="1" style="rounded=1;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#e2e8f0;strokeColor=#475569;strokeWidth=1;arcSize=15;" value="" vertex="1">
          <mxGeometry height="20" width="20" x="28" y="348" as="geometry" />
        </mxCell>
        <mxCell id="p-2" parent="1" style="rounded=1;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#e2e8f0;strokeColor=#475569;strokeWidth=1;arcSize=15;" value="" vertex="1">
          <mxGeometry height="20" width="20" x="53" y="348" as="geometry" />
        </mxCell>
        <mxCell id="p-3" parent="1" style="rounded=1;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#e2e8f0;strokeColor=#475569;strokeWidth=1;arcSize=15;" value="" vertex="1">
          <mxGeometry height="20" width="20" x="78" y="348" as="geometry" />
        </mxCell>
        <mxCell id="p-4" parent="1" style="rounded=1;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#e2e8f0;strokeColor=#475569;strokeWidth=1;arcSize=15;" value="" vertex="1">
          <mxGeometry height="20" width="20" x="103" y="348" as="geometry" />
        </mxCell>
        <mxCell id="p-5" parent="1" style="rounded=1;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#e2e8f0;strokeColor=#475569;strokeWidth=1;arcSize=15;" value="" vertex="1">
          <mxGeometry height="20" width="20" x="128" y="348" as="geometry" />
        </mxCell>
        <mxCell id="p-6" parent="1" style="rounded=1;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#e2e8f0;strokeColor=#475569;strokeWidth=1;arcSize=15;" value="" vertex="1">
          <mxGeometry height="20" width="20" x="153" y="348" as="geometry" />
        </mxCell>
        <mxCell id="p-lbl" parent="1" style="text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=center;verticalAlign=top;rounded=0;fontFamily=Times New Roman;fontSize=12;fontColor=#0f172a;" value="&lt;b&gt;PDK &amp;amp; Layout Parameters&lt;/b&gt;&lt;br&gt;&lt;i&gt;&lt;b&gt;p&lt;/b&gt;&lt;/i&gt; &amp;isin; &amp;#8477;&lt;sup&gt;&lt;i&gt;d&lt;/i&gt;&lt;/sup&gt;" vertex="1">
          <mxGeometry height="38" width="165" x="18" y="373" as="geometry" />
        </mxCell>

        <mxCell id="arr-p-cond" edge="1" parent="1" style="endArrow=block;endFill=1;endSize=5;html=1;rounded=0;strokeWidth=1.5;strokeColor=#475569;" value="">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="178" y="358" as="sourcePoint" />
            <mxPoint x="228" y="358" as="targetPoint" />
          </mxGeometry>
        </mxCell>

        <mxCell id="cond-mod" parent="1" style="swimlane;whiteSpace=wrap;html=1;fontFamily=Times New Roman;fillColor=#fffbeb;strokeColor=#f59e0b;strokeWidth=1.2;fontStyle=1;fontSize=13;fontColor=#78350f;startSize=26;rounded=1;arcSize=6;shadow=0;" value="PDK Conditioning Module" vertex="1">
          <mxGeometry height="153" width="290" x="230" y="284" as="geometry" />
        </mxCell>
        <mxCell id="opt-a" parent="cond-mod" style="rounded=1;whiteSpace=wrap;html=1;fontFamily=Times New Roman;fillColor=#ffffff;strokeColor=#fde68a;strokeWidth=1;fontSize=12;fontColor=#0f172a;arcSize=10;" value="&lt;b&gt;Option A: Concat&lt;/b&gt; &amp;nbsp;&lt;font color=&quot;#5c4033&quot;&gt;(Conv&lt;sub&gt;1&amp;times;1&lt;/sub&gt;([&lt;i&gt;F&lt;/i&gt;, &lt;b&gt;&lt;i&gt;p&lt;/i&gt;&lt;/b&gt;]))&lt;/font&gt;" vertex="1">
          <mxGeometry height="34" width="266" x="12" y="34" as="geometry" />
        </mxCell>
        <mxCell id="opt-b" parent="cond-mod" style="rounded=1;whiteSpace=wrap;html=1;fontFamily=Times New Roman;fillColor=#ffffff;strokeColor=#fde68a;strokeWidth=1;fontSize=12;fontColor=#0f172a;arcSize=10;" value="&lt;b&gt;Option B: FiLM&lt;/b&gt; &amp;nbsp;&lt;font color=&quot;#5c4033&quot;&gt;(&amp;gamma;(&lt;b&gt;&lt;i&gt;p&lt;/i&gt;&lt;/b&gt;) &amp;odot; &lt;i&gt;F&lt;/i&gt; + &amp;beta;(&lt;b&gt;&lt;i&gt;p&lt;/i&gt;&lt;/b&gt;))&lt;/font&gt;" vertex="1">
          <mxGeometry height="34" width="266" x="12" y="73" as="geometry" />
        </mxCell>
        <mxCell id="opt-c" parent="cond-mod" style="rounded=1;whiteSpace=wrap;html=1;fontFamily=Times New Roman;fillColor=#ffffff;strokeColor=#fde68a;strokeWidth=1;fontSize=12;fontColor=#0f172a;arcSize=10;" value="&lt;b&gt;Option C: Cross-Attention&lt;/b&gt; &amp;nbsp;&lt;font color=&quot;#5c4033&quot;&gt;(Spatial &lt;i&gt;Q&lt;/i&gt; &amp;times; PDK &lt;i&gt;K, V&lt;/i&gt;)&lt;/font&gt;" vertex="1">
          <mxGeometry height="34" width="266" x="12" y="112" as="geometry" />
        </mxCell>

        <mxCell id="arr-cond-mod" edge="1" parent="1" style="endArrow=block;endFill=1;endSize=5;html=1;rounded=0;strokeWidth=1.5;strokeColor=#d97706;" value="">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="375" y="284" as="sourcePoint" />
            <mxPoint x="375" y="201" as="targetPoint" />
          </mxGeometry>
        </mxCell>

      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''

# Проверяем валидность XML перед записью
ET.fromstring(final_xml)

with open("arch_DRC_ready.drawio.xml", "w", encoding="utf-8") as f:
    f.write(final_xml)

print("XML проверен и сохранён в arch_DRC_ready.drawio.xml")