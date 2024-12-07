const puppeteer = require('puppeteer');
const TARGET_URL = "https://dont-bet-on-sports-hp-test.streamlit.app/";
const WAKE_UP_BUTTON_TEXT = "Make me";
const PAGE_LOAD_GRACE_PERIOD_MS = 15000;

(async () => {
    const browser = await puppeteer.launch(
        { args: ["--no-sandbox"] }
    );

    const page = await browser.newPage();
    await page.goto(TARGET_URL);
    // Wait a grace period for the application to load
    await page.waitForTimeout(PAGE_LOAD_GRACE_PERIOD_MS);

    const checkForHibernation = async (target) => {
        // Look for any buttons containing the target text of the reboot button
        const [button] = await target.$x(`//button[contains(., '${WAKE_UP_BUTTON_TEXT}')]`);
        if (button) {
            console.log("Keeping app awake!");
            await button.click();
        }
    }

    await checkForHibernation(page);
    const frames = (await page.frames());
    for (const frame of frames) {
        await checkForHibernation(frame);
    }

    await browser.close();
})();
